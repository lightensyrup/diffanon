from model_anon import Trainer
from argparse import ArgumentParser


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--config_path",type=str,default="./config.json")
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--resume_milestone", type=int, default=None)
    args = parser.parse_args()
    trainer = Trainer(
        args.config_path,
        resume_dir=args.resume_dir,
        resume_milestone=args.resume_milestone,
    )
    trainer.train()
