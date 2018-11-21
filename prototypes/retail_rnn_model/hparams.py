hparams_manual = dict(
    train_window=60,
    batch_size=64,
    encoder_rnn_layers=1,
    decoder_rnn_layers=1,
    rnn_depth=400,
    encoder_dropout=0.03,
    gate_dropout=0.997,
    decoder_input_dropout=[1.0],
    decoder_state_dropout=[0.99],
    decoder_output_dropout=[0.975],
    decoder_variational_dropout=[False],
    asgd_decay=None,
    max_epoch=100
)

