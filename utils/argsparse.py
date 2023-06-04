import argparse


def get_args(jupyter_args = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--permute_truncated', required=False, action='store_true', 
                        help="Bool for activating permutation invariance")
    parser.add_argument('--retrain_seq2seq', required=False, action='store_true', 
                        help="Bool for retraining seq2seq module from scratch at each relabel round ")
    parser.add_argument('--use_prob_embedding', required=False, action='store_true', 
                        help="Bool for using original probability based embedding proposed in the original paper")
    parser.add_argument('--sequence_length', required=False, type=int, default=10, 
                        help="Length of truncated subsequences used in the seq2seq training")
    parser.add_argument('--rv', required=False, type=int, default=1, 
                        help="'round value' hyperparameter used for probability embedding, if activated")
    parser.add_argument('--ps_epochs', required=False, type=int, default=50, 
                        help="number of training epochs for per-step detectors")
    parser.add_argument('--relabel_rounds', required=False, type=int, default=1, 
                        help="Number of relabel rounds")
    parser.add_argument('--patience', required=False, type=int, default=None,
                        help="Patience for early stopping. Any value activates early stopping.")
    
    #------------maggie add parameters-------
    parser.add_argument('--seed', required=False, type=int, default=0,
                        help="random seed.")  
    parser.add_argument('--eps', required=False, type=float, default=1.0,
                        help="adversarial evasion attack parameter: epsilon.")        
    parser.add_argument('--eps_step', required=False, type=float, default=0.5,
                        help="adversarial evasion attack parameter: epsilon step.")   
    parser.add_argument('--max_iter', required=False, type=int, default=20,
                        help="adversarial evasion attack parameter: iteration number.")   
    parser.add_argument('--save_path',type=str, default='/home/huan1932/ARIoTEDef/result',help='Output path for saving results')
    parser.add_argument('--batchsize', required=False, type=int, default=32,
                        help="batch size.")       
    parser.add_argument('--timesteps', required=False, type=int, default=1,
                        help="time steps of LSTM.")      
    parser.add_argument('--seq2seq_epochs', required=False, type=int, default=50,
                        help="seq2seq epochs")      
        
    parser.add_argument('--retrainset_mode',type=str, default='adv',help='Output path for saving results')
        
        

    #----------------------------------------
    
    if jupyter_args is not None:
        args = parser.parse_args(jupyter_args)
    else: 
        args = parser.parse_args()
    return args
