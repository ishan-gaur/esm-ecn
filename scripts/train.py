"""
Train an MLP on the CLS or average pooled ESM embeddings for the ECN dataset.
"""
import fire
from esm_ecn.model import MLP, LitModel
from esm_ecn.data import train_data_loader, val_data_loader, test_data_loader
from esm_ecn.train import setup_experiment, load_best_checkpoint


def main(experiment_name=None, model_type="esmc_300m", cls=True, resume=False, wandb_debug=False, batch_size=2048, 
         epochs=1, test_only=False, accelerator="cpu", focal_loss=False, project="ecn", softmax=False, lr=0.0001):

    if test_only:
        resume = True

    train_loader = train_data_loader(model_type, batch_size, cls)
    val_loader = val_data_loader(model_type, batch_size, cls)
    test_loader = test_data_loader(model_type, batch_size, cls)

    input_dim = train_loader.dataset[0][0].shape[0]
    output_dim = train_loader.dataset[0][1].shape[0]
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    mlp = MLP(input_dim, output_dim=output_dim, sigmoid=(not softmax))
    model = LitModel(model=mlp, focal_loss=focal_loss, lr=lr)

    trainer, experiment_name = setup_experiment(experiment_name, project, resume, wandb_debug, accelerator, epochs)

    if resume:
        model = load_best_checkpoint(experiment_name, mlp, focal_loss, lr)

    if test_only:
        pass
    else:
        trainer.fit(model, train_loader, val_loader)

    model = load_best_checkpoint(experiment_name, mlp, focal_loss, lr)
    trainer.test(model, test_loader)

if __name__ == '__main__':
    fire.Fire(main)
