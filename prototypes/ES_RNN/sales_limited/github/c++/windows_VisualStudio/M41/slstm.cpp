/*
My implementation of dilated LSTMs, based on Dynet LSTM builders
- DilatedLSTMBuilder - standard Dilated LSTM (https://papers.nips.cc/paper/6613-dilated-recurrent-neural-networks.pdf)
- ResidualDilatedLSTMBuilder - Dilated LSTM with special Residual shortcuts, after https://arxiv.org/abs/1701.03360
- AttentiveDilatedLSTMBuilder - Dilated LSTM with Attention mechanism, as in the second stage of https://arxiv.org/abs/1704.02971
*
Slawek Smyl, Mar-May 2018
*/

#include "slstm.h"
#include "dynet/lstm.h"
#include "dynet/param-init.h"

#include <fstream>
#include <string>
#include <vector>
#include <iostream>

#if defined DEBUG
  #define _DEBUG
#endif

using namespace std;

namespace dynet {

  // ResidualDilatedLSTMBuilder based on Vanilla LSTM
  enum { _X2I, _H2I, _BI, _X2F, _H2F, _BF, _X2O, _H2O, _BO, _X2G, _H2G, _BG };
  enum { LN_GH, LN_BH, LN_GX, LN_BX, LN_GC, LN_BC };

  ResidualDilatedLSTMBuilder::ResidualDilatedLSTMBuilder() : has_initial_state(false), layers(0), input_dim(0), hid(0), dropout_rate_h(0), ln_lstm(false), forget_bias(1.f), dropout_masks_valid(false) { }

  ResidualDilatedLSTMBuilder::ResidualDilatedLSTMBuilder(vector<unsigned> dilations,
    unsigned input_dim,
    unsigned hidden_dim,
    ParameterCollection& model,
    bool ln_lstm, float forget_bias) : dilations(dilations), layers(unsigned(dilations.size())),
      input_dim(input_dim), hid(hidden_dim), ln_lstm(ln_lstm), forget_bias(forget_bias), dropout_masks_valid(false) {
    unsigned layer_input_dim = input_dim;
    local_model = model.add_subcollection("ResidualDilated-lstm-builder");
    for (unsigned i = 0; i < layers; ++i) {
      // i
      Parameter p_x2i = local_model.add_parameters({ hidden_dim * 4, layer_input_dim });
      Parameter p_h2i = local_model.add_parameters({ hidden_dim * 4, hidden_dim });
      //Parameter p_c2i = model.add_parameters({hidden_dim, hidden_dim});
      Parameter p_bi = local_model.add_parameters({ hidden_dim * 4 }, ParameterInitConst(0.f));

      layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

      vector<Parameter> ps = { p_x2i, p_h2i, /*p_c2i,*/ p_bi };
      params.push_back(ps);

      if (ln_lstm) {
        Parameter p_gh = model.add_parameters({ hidden_dim * 4 }, ParameterInitConst(1.f));
        Parameter p_bh = model.add_parameters({ hidden_dim * 4 }, ParameterInitConst(0.f));
        Parameter p_gx = model.add_parameters({ hidden_dim * 4 }, ParameterInitConst(1.f));
        Parameter p_bx = model.add_parameters({ hidden_dim * 4 }, ParameterInitConst(0.f));
        Parameter p_gc = model.add_parameters({ hidden_dim }, ParameterInitConst(1.f));
        Parameter p_bc = model.add_parameters({ hidden_dim }, ParameterInitConst(0.f));
        vector<Parameter> ln_ps = { p_gh, p_bh, p_gx, p_bx, p_gc, p_bc };
        ln_params.push_back(ln_ps);
      }
    }  // layers
    dropout_rate = 0.f;
    dropout_rate_h = 0.f;
  }

  void ResidualDilatedLSTMBuilder::new_graph_impl(ComputationGraph& cg, bool update) {
    param_vars.clear();
    if (ln_lstm)ln_param_vars.clear();
    for (unsigned i = 0; i < layers; ++i) {
      auto& p = params[i];
      vector<Expression> vars;
      for (unsigned j = 0; j < p.size(); ++j) { vars.push_back(update ? parameter(cg, p[j]) : const_parameter(cg, p[j])); }
      param_vars.push_back(vars);
      if (ln_lstm) {
        auto& ln_p = ln_params[i];
        vector<Expression> ln_vars;
        for (unsigned j = 0; j < ln_p.size(); ++j) { ln_vars.push_back(update ? parameter(cg, ln_p[j]) : const_parameter(cg, ln_p[j])); }
        ln_param_vars.push_back(ln_vars);
      }
    }

    _cg = &cg;
  }
  // layout: 0..layers = c
  //         layers+1..2*layers = h
  void ResidualDilatedLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
    h.clear();
    c.clear();

    if (hinit.size() > 0) {
      DYNET_ARG_CHECK(layers * 2 == hinit.size(),
        "ResidualDilatedLSTMBuilder must be initialized with 2 times as many expressions as layers "
        "(hidden state, and cell for each layer). However, for " << layers << " layers, " <<
        hinit.size() << " expressions were passed in");
      h0.resize(layers);
      c0.resize(layers);
      for (unsigned i = 0; i < layers; ++i) {
        c0[i] = hinit[i];
        h0[i] = hinit[i + layers];
      }
      has_initial_state = true;
    }
    else {
      has_initial_state = false;
    }

    dropout_masks_valid = false;
  }

  void ResidualDilatedLSTMBuilder::set_dropout_masks(unsigned batch_size) {
    masks.clear();
    for (unsigned i = 0; i < layers; ++i) {
      std::vector<Expression> masks_i;
      unsigned idim = (i == 0) ? input_dim : hid;
      if (dropout_rate > 0.f || dropout_rate_h > 0.f) {
        float retention_rate = 1.f - dropout_rate;
        float retention_rate_h = 1.f - dropout_rate_h;
        float scale = 1.f / retention_rate;
        float scale_h = 1.f / retention_rate_h;
        // in
        masks_i.push_back(random_bernoulli(*_cg, Dim({ idim }, batch_size), retention_rate, scale));
        // h
        masks_i.push_back(random_bernoulli(*_cg, Dim({ hid }, batch_size), retention_rate_h, scale_h));
        masks.push_back(masks_i);
      }
    }
    dropout_masks_valid = true;
  }

  ParameterCollection & ResidualDilatedLSTMBuilder::get_parameter_collection() {
    return local_model;
  }

  // TODO - Make this correct
  // Copied c from the previous step (otherwise c.size()< h.size())
  // Also is creating a new step something we want?
  // wouldn't overwriting the current one be better?
  Expression ResidualDilatedLSTMBuilder::set_h_impl(int prev, const vector<Expression>& h_new) {
    DYNET_ARG_CHECK(h_new.empty() || h_new.size() == layers,
      "ResidualDilatedLSTMBuilder::set_h expects as many inputs as layers, but got " <<
      h_new.size() << " inputs for " << layers << " layers");
    const unsigned t = h.size();
    h.push_back(vector<Expression>(layers));
    c.push_back(vector<Expression>(layers));
    for (unsigned i = 0; i < layers; ++i) {
      Expression h_i = h_new[i];
      Expression c_i = c[t - 1][i];
      h[t][i] = h_i;
      c[t][i] = c_i;
    }
    return h[t].back();
  }
  // Current implementation : s_new is either {new_c[0],...,new_c[n]}
  // or {new_c[0],...,new_c[n],new_h[0],...,new_h[n]}
  Expression ResidualDilatedLSTMBuilder::set_s_impl(int prev, const std::vector<Expression>& s_new) {
    DYNET_ARG_CHECK(s_new.size() == layers || s_new.size() == 2 * layers,
      "ResidualDilatedLSTMBuilder::set_s expects either as many inputs or twice as many inputs as layers, but got " << s_new.size() << " inputs for " << layers << " layers");
    bool only_c = s_new.size() == layers;
    const unsigned t = c.size();
    h.push_back(vector<Expression>(layers));
    c.push_back(vector<Expression>(layers));
    for (unsigned i = 0; i < layers; ++i) {
      Expression h_i = only_c ? h[t - 1][i] : s_new[i + layers];
      Expression c_i = s_new[i];
      h[t][i] = h_i;
      c[t][i] = c_i;
    }
    return h[t].back();
  }

  Expression ResidualDilatedLSTMBuilder::add_input_impl(int prev, const Expression& x) {
    h.push_back(vector<Expression>(layers));
    c.push_back(vector<Expression>(layers));
    vector<Expression>& ht = h.back();
    vector<Expression>& ct = c.back();
    Expression in = x;
    if ((dropout_rate > 0.f || dropout_rate_h > 0.f) && !dropout_masks_valid) set_dropout_masks(x.dim().bd);
    for (unsigned i = 0; i < layers; ++i) {
    	int dilation_offset = dilations[i] - 1;
      const vector<Expression>& vars = param_vars[i];

      Expression i_h_tm1, i_c_tm1;
      bool has_prev_state = (prev >= 0 || has_initial_state);
      if (prev < dilation_offset) {
        if (has_initial_state) {
          // intial value for h and c at timestep 0 in layer i
          // defaults to zero matrix input if not set in add_parameter_edges
          i_h_tm1 = h0[i];
          i_c_tm1 = c0[i];
        }
        else {
          i_h_tm1 = zeros(*_cg, Dim({ vars[_BI].dim()[0] / 4 }, x.dim().bd));
          i_c_tm1 = i_h_tm1;
        }
      }
      else {
        i_h_tm1 = h[prev - dilation_offset][i];
        i_c_tm1 = c[prev - dilation_offset][i];
      }
      // apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
      if (dropout_rate > 0.f) {
        in = cmult(in, masks[i][0]);
      }
      if (has_prev_state && dropout_rate_h > 0.f)
        i_h_tm1 = cmult(i_h_tm1, masks[i][1]);
      // input
      Expression tmp;
      Expression i_ait;
      Expression i_aft;
      Expression i_aot;
      Expression i_agt;
      if (ln_lstm) {
        const vector<Expression>& ln_vars = ln_param_vars[i];
        if (has_prev_state)
          tmp = vars[_BI] + layer_norm(vars[_X2I] * in, ln_vars[LN_GX], ln_vars[LN_BX]) + layer_norm(vars[_H2I] * i_h_tm1, ln_vars[LN_GH], ln_vars[LN_BH]);
        else
          tmp = vars[_BI] + layer_norm(vars[_X2I] * in, ln_vars[LN_GX], ln_vars[LN_BX]);
      }
      else {
        if (has_prev_state)
          tmp = affine_transform({ vars[_BI], vars[_X2I], in, vars[_H2I], i_h_tm1 });
        else
          tmp = affine_transform({ vars[_BI], vars[_X2I], in });
      }
      i_ait = pick_range(tmp, 0, hid);
      i_aft = pick_range(tmp, hid, hid * 2);
      i_aot = pick_range(tmp, hid * 2, hid * 3);
      i_agt = pick_range(tmp, hid * 3, hid * 4);
      Expression i_it = logistic(i_ait);
      if (forget_bias != 0.0)
        tmp = logistic(i_aft + forget_bias);
      else
        tmp = logistic(i_aft);

      Expression i_ft = tmp;
      Expression i_ot = logistic(i_aot);
      Expression i_gt = tanh(i_agt);

      ct[i] = has_prev_state ? (cmult(i_ft, i_c_tm1) + cmult(i_it, i_gt)) : cmult(i_it, i_gt);
      if (ln_lstm) {
        const vector<Expression>& ln_vars = ln_param_vars[i];
        if (i==0)
        	in = ht[i] = cmult(i_ot, tanh(layer_norm(ct[i], ln_vars[LN_GC], ln_vars[LN_BC])));
        else
        	in = ht[i] = cmult(i_ot, in+tanh(layer_norm(ct[i], ln_vars[LN_GC], ln_vars[LN_BC])));
      }
      else  {
      	if (i==0)
          in = ht[i] = cmult(i_ot, tanh(ct[i]));
      	else
      		in = ht[i] = cmult(i_ot, in+tanh(ct[i]));
      }
    }
    return ht.back();
  }

  void ResidualDilatedLSTMBuilder::copy(const RNNBuilder & rnn) {
    const ResidualDilatedLSTMBuilder & rnn_lstm = (const ResidualDilatedLSTMBuilder&)rnn;
    DYNET_ARG_CHECK(params.size() == rnn_lstm.params.size(),
      "Attempt to copy ResidualDilatedLSTMBuilder with different number of parameters "
      "(" << params.size() << " != " << rnn_lstm.params.size() << ")");
    for (size_t i = 0; i < params.size(); ++i)
      for (size_t j = 0; j < params[i].size(); ++j)
        params[i][j] = rnn_lstm.params[i][j];
    for (size_t i = 0; i < ln_params.size(); ++i)
      for (size_t j = 0; j < ln_params[i].size(); ++j)
        ln_params[i][j] = rnn_lstm.ln_params[i][j];
  }

  void ResidualDilatedLSTMBuilder::set_dropout(float d) {
    DYNET_ARG_CHECK(d >= 0.f && d <= 1.f,
      "dropout rate must be a probability (>=0 and <=1)");
    dropout_rate = d;
    dropout_rate_h = d;
  }

  void ResidualDilatedLSTMBuilder::set_dropout(float d, float d_h) {
    DYNET_ARG_CHECK(d >= 0.f && d <= 1.f && d_h >= 0.f && d_h <= 1.f,
      "dropout rate must be a probability (>=0 and <=1)");
    dropout_rate = d;
    dropout_rate_h = d_h;
  }

  void ResidualDilatedLSTMBuilder::disable_dropout() {
    dropout_rate = 0.f;
    dropout_rate_h = 0.f;
  }




  //enum { _X2I, _H2I, _BI, _X2F, _H2F, _BF, _X2O, _H2O, _BO, _X2G, _H2G, _BG };
  enum { _X2I_, _H2I_, _BI_, _XA1, _HA1, _SA1, _BA1, _A2, _B2 };


//***************************


  
  AttentiveDilatedLSTMBuilder::AttentiveDilatedLSTMBuilder() : has_initial_state(false), layers(0), input_dim(0), hid(0), dropout_rate_h(0), weightnoise_std(0), dropout_masks_valid(false) { }
  
  AttentiveDilatedLSTMBuilder::AttentiveDilatedLSTMBuilder(vector<unsigned> max_dilations,
                                         unsigned input_dim,
                                         unsigned hidden_dim,
                                         unsigned attention_dim,
                                         ParameterCollection& model)
  : max_dilations(max_dilations), layers(unsigned(max_dilations.size())),
    input_dim(input_dim), hid(hidden_dim), attention_dim(attention_dim), weightnoise_std(0), dropout_masks_valid(false) {
    unsigned layer_input_dim = input_dim;
    local_model = model.add_subcollection("compact-vanilla-lstm-builder");
    for (unsigned i = 0; i < layers; ++i) {
      // i
      Parameter p_Wx = local_model.add_parameters({ hidden_dim * 4, layer_input_dim });
      Parameter p_Wh = local_model.add_parameters({ hidden_dim * 4, hidden_dim });
      Parameter p_b = local_model.add_parameters({ hidden_dim * 4 }, ParameterInitConst(0.f));
      
      Parameter p_Wxa1 = local_model.add_parameters({ attention_dim, layer_input_dim });
      Parameter p_Wha1 = local_model.add_parameters({ attention_dim, hidden_dim });
      Parameter p_Wsa1 = local_model.add_parameters({ attention_dim, hidden_dim });
      Parameter p_ba1 = local_model.add_parameters({ attention_dim }, ParameterInitConst(0.f));
      
      Parameter p_Wa2 = local_model.add_parameters({ max_dilations[i], attention_dim });
      Parameter p_ba2 = local_model.add_parameters({ max_dilations[i] }, ParameterInitConst(0.f));
      
      layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next
      
      vector<Parameter> ps = { p_Wx, p_Wh, p_b, p_Wxa1, p_Wha1, p_Wsa1, p_ba1, p_Wa2, p_ba2 };
      params.push_back(ps);
      
    }  // layers
    dropout_rate = 0.f;
    dropout_rate_h = 0.f;
  }
  
  void AttentiveDilatedLSTMBuilder::new_graph_impl(ComputationGraph& cg, bool update) {
    param_vars.clear();
    for (unsigned i = 0; i < layers; ++i) {
      auto& p = params[i];
      vector<Expression> vars;
      for (unsigned j = 0; j < p.size(); ++j) { 
        vars.push_back(update ? parameter(cg, p[j]) : const_parameter(cg, p[j])); 
      }
      param_vars.push_back(vars);
    }
    
    _cg = &cg;
  }
  // layout: 0..layers = c
  //         layers+1..2*layers = h
  void AttentiveDilatedLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
    h.clear();
    c.clear();
    
    if (hinit.size() > 0) {
      DYNET_ARG_CHECK(layers * 2 == hinit.size(),
                      "AttentiveDilatedLSTMBuilder must be initialized with 2 times as many expressions as layers "
                      "(hidden state, and cell for each layer). However, for " << layers << " layers, " <<
                      hinit.size() << " expressions were passed in");
      h0.resize(layers);
      c0.resize(layers);
      for (unsigned i = 0; i < layers; ++i) {
        c0[i] = hinit[i];
        h0[i] = hinit[i + layers];
      }
      has_initial_state = true;
    }
    else {
      has_initial_state = false;
    }
    
    dropout_masks_valid = false;
  }
  
  void AttentiveDilatedLSTMBuilder::set_dropout_masks(unsigned batch_size) {
    masks.clear();
    for (unsigned i = 0; i < layers; ++i) {
      std::vector<Expression> masks_i;
      unsigned idim = (i == 0) ? input_dim : hid;
      if (dropout_rate > 0.f || dropout_rate_h > 0.f) {
        float retention_rate = 1.f - dropout_rate;
        float retention_rate_h = 1.f - dropout_rate_h;
        float scale = 1.f / retention_rate;
        float scale_h = 1.f / retention_rate_h;
        // in
        masks_i.push_back(random_bernoulli(*_cg, Dim({ idim }, batch_size), retention_rate, scale));
        // h
        masks_i.push_back(random_bernoulli(*_cg, Dim({ hid }, batch_size), retention_rate_h, scale_h));
        masks.push_back(masks_i);
      }
    }
    dropout_masks_valid = true;
  }
  
  ParameterCollection & AttentiveDilatedLSTMBuilder::get_parameter_collection() {
    return local_model;
  }
  
  // TODO - Make this correct
  // Copied c from the previous step (otherwise c.size()< h.size())
  // Also is creating a new step something we want?
  // wouldn't overwriting the current one be better?
  Expression AttentiveDilatedLSTMBuilder::set_h_impl(int prev, const vector<Expression>& h_new) {
    DYNET_ARG_CHECK(h_new.empty() || h_new.size() == layers,
                    "AttentiveDilatedLSTMBuilder::set_h expects as many inputs as layers, but got " <<
                    h_new.size() << " inputs for " << layers << " layers");
    const unsigned t = unsigned(h.size());
    h.push_back(vector<Expression>(layers));
    c.push_back(vector<Expression>(layers));
    for (unsigned i = 0; i < layers; ++i) {
      Expression h_i = h_new[i];
      Expression c_i = c[t - 1][i];
      h[t][i] = h_i;
      c[t][i] = c_i;
    }
    return h[t].back();
  }
  // Current implementation : s_new is either {new_c[0],...,new_c[n]}
  // or {new_c[0],...,new_c[n],new_h[0],...,new_h[n]}
  Expression AttentiveDilatedLSTMBuilder::set_s_impl(int prev, const std::vector<Expression>& s_new) {
    DYNET_ARG_CHECK(s_new.size() == layers || s_new.size() == 2 * layers,
                    "AttentiveDilatedLSTMBuilder::set_s expects either as many inputs or twice as many inputs as layers, but got " << s_new.size() << " inputs for " << layers << " layers");
    bool only_c = s_new.size() == layers;
    const unsigned t = unsigned(c.size());
    h.push_back(vector<Expression>(layers));
    c.push_back(vector<Expression>(layers));
    for (unsigned i = 0; i < layers; ++i) {
      Expression h_i = only_c ? h[t - 1][i] : s_new[i + layers];
      Expression c_i = s_new[i];
      h[t][i] = h_i;
      c[t][i] = c_i;
    }
    return h[t].back();
  }
  
  Expression AttentiveDilatedLSTMBuilder::add_input_impl(int prev, const Expression& x) {
    h.push_back(vector<Expression>(layers));
    c.push_back(vector<Expression>(layers));
    vector<Expression>& ht = h.back();
    vector<Expression>& ct = c.back();
    Expression in = x;
    if ((dropout_rate > 0.f || dropout_rate_h > 0.f) && !dropout_masks_valid) set_dropout_masks(x.dim().bd);
    for (unsigned i = 0; i < layers; ++i) {
      int dilation_offset= max_dilations[i]-1;
      const vector<Expression>& vars = param_vars[i];
      Expression i_h_tm1, i_c_tm1;
      if (prev < dilation_offset) {
        if (has_initial_state) {
          // initial value for h and c at timestep 0 in layer i
          // defaults to zero matrix input if not set in add_parameter_edges
          i_h_tm1 = h0[i];
          i_c_tm1 = c0[i];
        }
        else {
          i_h_tm1 = zeros(*_cg, Dim({ vars[_BI].dim()[0] / 4 }, x.dim().bd));
          i_c_tm1 = i_h_tm1;
        }
      }
      else {
        if (dilation_offset>0) {
          //enum { _X2I, _H2I, _BI, _XA1, _HA1, _SA1, _BA1, _A2, _B2 };
          Expression weights_ex=vars[_XA1]*in+ vars[_HA1]*h[prev][i]+ vars[_SA1]*c[prev][i]+ vars[_BA1];
          weights_ex=tanh(weights_ex);
          weights_ex=vars[_A2]* weights_ex+ vars[_B2];
          weights_ex =softmax(weights_ex);
          #if defined _DEBUG
            vector<float> weights=as_vector(weights_ex.value());
          #endif

          unsigned indx=0;
          Expression w_ex = pick(weights_ex, indx);
          Expression avg_h= cmult(h[prev][i], w_ex);
          for (indx=1; indx <= dilation_offset; indx++) {//dilation_offset==max_dilations[i]-1, so together with indx==0, we cover max_dilations[i] steps
            w_ex = pick(weights_ex, indx);
            avg_h = avg_h+cmult(h[prev- indx][i], w_ex);
          }
          i_h_tm1 = avg_h;
        } else {
          i_h_tm1 = h[prev- dilation_offset][i];
        }
        i_c_tm1 = c[prev- dilation_offset][i];
      }
      if (dropout_rate > 0.f || dropout_rate_h > 0.f) {
        // apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
        Expression gates_t = vanilla_lstm_gates_dropout({ in }, i_h_tm1, vars[_X2I], vars[_H2I], vars[_BI], masks[i][0], masks[i][1], weightnoise_std);
        ct[i] = vanilla_lstm_c(i_c_tm1, gates_t);
        in = ht[i] = vanilla_lstm_h(ct[i], gates_t);
      }
      else {
        Expression gates_t = vanilla_lstm_gates({ in }, i_h_tm1, vars[_X2I], vars[_H2I], vars[_BI], weightnoise_std);
        ct[i] = vanilla_lstm_c(i_c_tm1, gates_t);
        in = ht[i] = vanilla_lstm_h(ct[i], gates_t);
      }
    }
    return ht.back();
  }
  
  void AttentiveDilatedLSTMBuilder::copy(const RNNBuilder & rnn) {
    const AttentiveDilatedLSTMBuilder & rnn_lstm = (const AttentiveDilatedLSTMBuilder&)rnn;
    DYNET_ARG_CHECK(params.size() == rnn_lstm.params.size(),
                    "Attempt to copy AttentiveDilatedLSTMBuilder with different number of parameters "
                    "(" << params.size() << " != " << rnn_lstm.params.size() << ")");
    for (size_t i = 0; i < params.size(); ++i)
      for (size_t j = 0; j < params[i].size(); ++j)
        params[i][j] = rnn_lstm.params[i][j];
  }
  
  void AttentiveDilatedLSTMBuilder::set_dropout(float d) {
    DYNET_ARG_CHECK(d >= 0.f && d <= 1.f,
                    "dropout rate must be a probability (>=0 and <=1)");
    dropout_rate = d;
    dropout_rate_h = d;
  }
  
  void AttentiveDilatedLSTMBuilder::set_dropout(float d, float d_h) {
    DYNET_ARG_CHECK(d >= 0.f && d <= 1.f && d_h >= 0.f && d_h <= 1.f,
                    "dropout rate must be a probability (>=0 and <=1)");
    dropout_rate = d;
    dropout_rate_h = d_h;
  }
  
  void AttentiveDilatedLSTMBuilder::disable_dropout() {
    dropout_rate = 0.f;
    dropout_rate_h = 0.f;
  }
  void AttentiveDilatedLSTMBuilder::set_weightnoise(float std) {
    DYNET_ARG_CHECK(std >= 0.f, "weight noise must have standard deviation >=0");
    weightnoise_std = std;
  }

  //*/

  DilatedLSTMBuilder::DilatedLSTMBuilder() : has_initial_state(false), layers(0), input_dim(0), hid(0), dropout_rate_h(0), weightnoise_std(0), dropout_masks_valid(false) { }

  DilatedLSTMBuilder::DilatedLSTMBuilder(vector<unsigned> dilations,
    unsigned input_dim,
    unsigned hidden_dim,
    ParameterCollection& model)
    : dilations(dilations), layers(unsigned(dilations.size())),
    input_dim(input_dim), hid(hidden_dim), weightnoise_std(0), dropout_masks_valid(false) {
    unsigned layer_input_dim = input_dim;
    local_model = model.add_subcollection("compact-vanilla-lstm-builder");
    for (unsigned i = 0; i < layers; ++i) {
      // i
      Parameter p_Wx = local_model.add_parameters({ hidden_dim * 4, layer_input_dim });
      Parameter p_Wh = local_model.add_parameters({ hidden_dim * 4, hidden_dim });
      Parameter p_b = local_model.add_parameters({ hidden_dim * 4 }, ParameterInitConst(0.f));

      layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

      vector<Parameter> ps = { p_Wx, p_Wh, p_b };
      params.push_back(ps);

    }  // layers
    dropout_rate = 0.f;
    dropout_rate_h = 0.f;
  }

  void DilatedLSTMBuilder::new_graph_impl(ComputationGraph& cg, bool update) {
    param_vars.clear();
    for (unsigned i = 0; i < layers; ++i) {
      auto& p = params[i];
      vector<Expression> vars;
      for (unsigned j = 0; j < p.size(); ++j) { vars.push_back(update ? parameter(cg, p[j]) : const_parameter(cg, p[j])); }
      param_vars.push_back(vars);
    }

    _cg = &cg;
  }
  // layout: 0..layers = c
  //         layers+1..2*layers = h
  void DilatedLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
    h.clear();
    c.clear();

    if (hinit.size() > 0) {
      DYNET_ARG_CHECK(layers * 2 == hinit.size(),
        "DilatedLSTMBuilder must be initialized with 2 times as many expressions as layers "
        "(hidden state, and cell for each layer). However, for " << layers << " layers, " <<
        hinit.size() << " expressions were passed in");
      h0.resize(layers);
      c0.resize(layers);
      for (unsigned i = 0; i < layers; ++i) {
        c0[i] = hinit[i];
        h0[i] = hinit[i + layers];
      }
      has_initial_state = true;
    } else {
      has_initial_state = false;
    }

    dropout_masks_valid = false;
  }

  void DilatedLSTMBuilder::set_dropout_masks(unsigned batch_size) {
    masks.clear();
    for (unsigned i = 0; i < layers; ++i) {
      std::vector<Expression> masks_i;
      unsigned idim = (i == 0) ? input_dim : hid;
      if (dropout_rate > 0.f || dropout_rate_h > 0.f) {
        float retention_rate = 1.f - dropout_rate;
        float retention_rate_h = 1.f - dropout_rate_h;
        float scale = 1.f / retention_rate;
        float scale_h = 1.f / retention_rate_h;
        // in
        masks_i.push_back(random_bernoulli(*_cg, Dim({ idim }, batch_size), retention_rate, scale));
        // h
        masks_i.push_back(random_bernoulli(*_cg, Dim({ hid }, batch_size), retention_rate_h, scale_h));
        masks.push_back(masks_i);
      }
    }
    dropout_masks_valid = true;
  }

  ParameterCollection & DilatedLSTMBuilder::get_parameter_collection() {
    return local_model;
  }

  // TODO - Make this correct
  // Copied c from the previous step (otherwise c.size()< h.size())
  // Also is creating a new step something we want?
  // wouldn't overwriting the current one be better?
  Expression DilatedLSTMBuilder::set_h_impl(int prev, const vector<Expression>& h_new) {
    DYNET_ARG_CHECK(h_new.empty() || h_new.size() == layers,
      "DilatedLSTMBuilder::set_h expects as many inputs as layers, but got " <<
      h_new.size() << " inputs for " << layers << " layers");
    const unsigned t = unsigned(h.size());
    h.push_back(vector<Expression>(layers));
    c.push_back(vector<Expression>(layers));
    for (unsigned i = 0; i < layers; ++i) {
      Expression h_i = h_new[i];
      Expression c_i = c[t - 1][i];
      h[t][i] = h_i;
      c[t][i] = c_i;
    }
    return h[t].back();
  }
  // Current implementation : s_new is either {new_c[0],...,new_c[n]}
  // or {new_c[0],...,new_c[n],new_h[0],...,new_h[n]}
  Expression DilatedLSTMBuilder::set_s_impl(int prev, const std::vector<Expression>& s_new) {
    DYNET_ARG_CHECK(s_new.size() == layers || s_new.size() == 2 * layers,
      "DilatedLSTMBuilder::set_s expects either as many inputs or twice as many inputs as layers, but got " << s_new.size() << " inputs for " << layers << " layers");
    bool only_c = s_new.size() == layers;
    const unsigned t = unsigned(c.size());
    h.push_back(vector<Expression>(layers));
    c.push_back(vector<Expression>(layers));
    for (unsigned i = 0; i < layers; ++i) {
      Expression h_i = only_c ? h[t - 1][i] : s_new[i + layers];
      Expression c_i = s_new[i];
      h[t][i] = h_i;
      c[t][i] = c_i;
    }
    return h[t].back();
  }

  Expression DilatedLSTMBuilder::add_input_impl(int prev, const Expression& x) {
    h.push_back(vector<Expression>(layers));
    c.push_back(vector<Expression>(layers));
    vector<Expression>& ht = h.back();
    vector<Expression>& ct = c.back();
    Expression in = x;
    if ((dropout_rate > 0.f || dropout_rate_h > 0.f) && !dropout_masks_valid) set_dropout_masks(x.dim().bd);
    for (unsigned i = 0; i < layers; ++i) {
      int dilation_offset = dilations[i] - 1;
      const vector<Expression>& vars = param_vars[i];
      Expression i_h_tm1, i_c_tm1;
      if (prev < dilation_offset) {
        if (has_initial_state) {
          // initial value for h and c at timestep 0 in layer i
          // defaults to zero matrix input if not set in add_parameter_edges
          i_h_tm1 = h0[i];
          i_c_tm1 = c0[i];
        } else {
          i_h_tm1 = zeros(*_cg, Dim({ vars[_BI].dim()[0] / 4 }, x.dim().bd));
          i_c_tm1 = i_h_tm1;
        }
      } else {  // t > 0
        i_h_tm1 = h[prev - dilation_offset][i];
        i_c_tm1 = c[prev - dilation_offset][i];
      }
      if (dropout_rate > 0.f || dropout_rate_h > 0.f) {
        // apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
        Expression gates_t = vanilla_lstm_gates_dropout({ in }, i_h_tm1, vars[_X2I], vars[_H2I], vars[_BI], masks[i][0], masks[i][1], weightnoise_std);
        ct[i] = vanilla_lstm_c(i_c_tm1, gates_t);
        in = ht[i] = vanilla_lstm_h(ct[i], gates_t);
      } else {
        Expression gates_t = vanilla_lstm_gates({ in }, i_h_tm1, vars[_X2I], vars[_H2I], vars[_BI], weightnoise_std);
        ct[i] = vanilla_lstm_c(i_c_tm1, gates_t);
        in = ht[i] = vanilla_lstm_h(ct[i], gates_t);
      }
    }
    return ht.back();
  }

  void DilatedLSTMBuilder::copy(const RNNBuilder & rnn) {
    const DilatedLSTMBuilder & rnn_lstm = (const DilatedLSTMBuilder&)rnn;
    DYNET_ARG_CHECK(params.size() == rnn_lstm.params.size(),
      "Attempt to copy DilatedLSTMBuilder with different number of parameters "
      "(" << params.size() << " != " << rnn_lstm.params.size() << ")");
    for (size_t i = 0; i < params.size(); ++i)
      for (size_t j = 0; j < params[i].size(); ++j)
        params[i][j] = rnn_lstm.params[i][j];
  }

  void DilatedLSTMBuilder::set_dropout(float d) {
    DYNET_ARG_CHECK(d >= 0.f && d <= 1.f,
      "dropout rate must be a probability (>=0 and <=1)");
    dropout_rate = d;
    dropout_rate_h = d;
  }

  void DilatedLSTMBuilder::set_dropout(float d, float d_h) {
    DYNET_ARG_CHECK(d >= 0.f && d <= 1.f && d_h >= 0.f && d_h <= 1.f,
      "dropout rate must be a probability (>=0 and <=1)");
    dropout_rate = d;
    dropout_rate_h = d_h;
  }

  void DilatedLSTMBuilder::disable_dropout() {
    dropout_rate = 0.f;
    dropout_rate_h = 0.f;
  }
  void DilatedLSTMBuilder::set_weightnoise(float std) {
    DYNET_ARG_CHECK(std >= 0.f, "weight noise must have standard deviation >=0");
    weightnoise_std = std;
  }

} // namespace dynet
