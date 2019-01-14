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
    max_epoch=20,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08
)

hparams_smac = dict(
    train_window=26,
    batch_size=64,
    encoder_rnn_layers=1,
    decoder_rnn_layers=1,
    rnn_depth=387,
    encoder_dropout=0.024688459483309007,
    gate_dropout=0.980832247298109,
    decoder_input_dropout=[0.9975650671957902],
    decoder_state_dropout=[0.9743711264734845],
    decoder_output_dropout=[0.9732177111192211],
    decoder_variational_dropout=[False],
    asgd_decay=None,
    max_epoch=100,
    learning_rate=0.001,
    beta1=0.7763754022206656,
    beta2=0.7923825287287111,
    epsilon=1e-08
)

# this turns out to leads to overfitting on the validation data set
# MAPE on validation dataset: ~34%
# MAPE on test dataset: ~44%
hparams_smac_100 = dict(
    train_window=52,
    batch_size=256,
    encoder_rnn_layers=1,
    decoder_rnn_layers=1,
    rnn_depth=455,
    encoder_dropout=0.0040379628855595154,
    gate_dropout=0.9704657028012964,
    decoder_input_dropout=[0.9706046837200847],
    decoder_state_dropout=[0.9853308617869989],
    decoder_output_dropout=[0.9779977163697378],
    decoder_variational_dropout=[False],
    asgd_decay=None,
    max_epoch=200,
    learning_rate=0.01,
    beta1=0.6011027681578323,
    beta2=0.9809964662293627,
    epsilon=1e-08
)