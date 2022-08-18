from cytoself.datamanager.utils.test.test_splitdata_on_fov import gen_label

n_fovs = 40
test_label = gen_label(n_fovs)


def add_default_model_args(model_args):
    model_args['encoder_args'] = {
        'in_channels': model_args['input_shape'][0],
        'blocks_args': [
            {
                'expand_ratio': 1,
                'kernel': 3,
                'stride': 1,
                'input_channels': 32,
                'out_channels': 16,
                'num_layers': 1,
            }
        ],
        'out_channels': model_args['emb_shape'][0],
    }
    model_args['decoder_args'] = {
        'input_shape': model_args['emb_shape'],
        'num_residual_layers': 1,
        'output_shape': model_args['input_shape'],
    }
    return model_args


CYTOSELF_MODEL_ARGS = {
    'input_shape': (1, 32, 32),
    'emb_shapes': ((16, 16), (16, 16)),
    'output_shape': (1, 32, 32),
    'vq_args': {'num_embeddings': 7, 'embedding_dim': 16},
    'num_class': 3,
    'encoder_args': [
        {
            'blocks_args': [
                {
                    'expand_ratio': 1,
                    'kernel': 3,
                    'stride': 1,
                    'input_channels': 32,
                    'out_channels': 16,
                    'num_layers': 1,
                }
            ],
        },
        {
            'blocks_args': [
                {
                    'expand_ratio': 1,
                    'kernel': 3,
                    'stride': 1,
                    'input_channels': 32,
                    'out_channels': 16,
                    'num_layers': 1,
                }
            ],
        },
    ],
    'decoder_args': [{'num_residual_layers': 1}, {'num_residual_layers': 1}],
}
