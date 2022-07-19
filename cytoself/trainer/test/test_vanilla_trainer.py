from os.path import join, exists

import torch


def test_vanilla_ae_trainer_fit(vanilla_ae_trainer, opencell_datamgr_vanilla, basepath):
    vanilla_ae_trainer.fit(opencell_datamgr_vanilla, tensorboard_path=join(basepath, 'tb_log'))
    assert len(vanilla_ae_trainer.losses['train_loss']) == vanilla_ae_trainer.train_args['max_epochs']
    assert min(vanilla_ae_trainer.losses['train_loss']) < torch.inf


def test_vanilla_ae_trainer__detach_graph(vanilla_ae_trainer, opencell_datamgr_vanilla):
    _batch = next(iter(opencell_datamgr_vanilla.test_loader))
    timg = vanilla_ae_trainer._get_data_by_name(_batch, 'image')
    out = vanilla_ae_trainer.model(timg)
    loss = vanilla_ae_trainer.calc_loss_one_batch(out, torch.ones_like(timg).to(vanilla_ae_trainer.device))
    assert loss[0].requires_grad
    assert vanilla_ae_trainer._detach_graph(loss[0]).requires_grad is False
    assert vanilla_ae_trainer._detach_graph(loss[:1])[0].requires_grad is False


def test_infer_reconstruction(vanilla_ae_trainer, opencell_datamgr_vanilla):
    out = vanilla_ae_trainer.infer_reconstruction(opencell_datamgr_vanilla.test_loader)
    assert out.shape == opencell_datamgr_vanilla.test_dataset.data.shape

    d = next(iter(opencell_datamgr_vanilla.test_loader))
    out = vanilla_ae_trainer.infer_reconstruction(d['image'].numpy())
    assert out.shape == (opencell_datamgr_vanilla.batch_size,) + opencell_datamgr_vanilla.test_dataset.data.shape[1:]


def test_save_load_model(vanilla_ae_trainer):
    vanilla_ae_trainer.save_model(vanilla_ae_trainer.savepath_dict['homepath'], 'test.pt')
    assert exists(join(vanilla_ae_trainer.savepath_dict['homepath'], 'test.pt')), 'pytorch model was not found.'
    w = vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight
    vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight = torch.nn.Parameter(torch.zeros_like(w))
    vanilla_ae_trainer.load_model(join(vanilla_ae_trainer.savepath_dict['homepath'], 'test.pt'))
    assert (vanilla_ae_trainer.model.decoder.decoder.resrep1last.conv.weight == w).all()
