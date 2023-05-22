
from models.keraslstm import PSDetector

from seq2seq.seq2seq_attention import Seq2seqAttention

from models.keraslstm import Seq2Seq


def init_psdetector(multistep_dataset, args):

    reconnaissance_detector = PSDetector(name="reconnaissance-detector", args=args)
    reconnaissance_detector.add_dataset(dataset=multistep_dataset['reconnaissance']) 
    print("reconnaissance_detector.dataset['train'][0].shape:",reconnaissance_detector.dataset['train'][0].shape)
    print("reconnaissance_detector.dataset['train'][1].shape:",reconnaissance_detector.dataset['train'][1].shape)
    """
    reconnaissance_detector.dataset['train'][0].shape: (14400, 41)
    reconnaissance_detector.dataset['train'][1].shape: (14400,)
    """
    reconnaissance_input_dim=reconnaissance_detector.dataset['train'][0].shape[1]
    print("reconnaissance_input_dim:",reconnaissance_input_dim)               # reconnaissance_input_dim: 41
    # reconnaissance_detector.def_model(reconnaissance_input_dim, output_dim=1, timesteps=1)
    reconnaissance_detector.def_model(reconnaissance_input_dim, output_dim=1, timesteps=args.timesteps)


    infection_detector = PSDetector(name="infection-detector", args=args)
    infection_detector.add_dataset(dataset=multistep_dataset['infection']) 
    print("infection_detector.dataset['train'][0].shape:",infection_detector.dataset['train'][0].shape)
    print("infection_detector.dataset['train'][1].shape:",infection_detector.dataset['train'][1].shape)
    """
    infection_detector.dataset['train'][0].shape: (19808, 41)
    infection_detector.dataset['train'][1].shape: (19808,)
    """
    infection_input_dim=infection_detector.dataset['train'][0].shape[1]
    print("infection_input_dim:",infection_input_dim)               # infection_input_dim: 41
    # infection_detector.def_model(infection_input_dim, output_dim=1, timesteps=1)
    infection_detector.def_model(infection_input_dim, output_dim=1, timesteps=args.timesteps)

    attack_detector = PSDetector(name="attack-detector", args=args)
    attack_detector.add_dataset(dataset=multistep_dataset['attack']) 
    print("attack_detector.dataset['train'][0].shape:",attack_detector.dataset['train'][0].shape)
    print("attack_detector.dataset['train'][1].shape:",attack_detector.dataset['train'][1].shape)
    """
    attack_detector.dataset['train'][0].shape: (19136, 41)
    attack_detector.dataset['train'][1].shape: (19136,)
    """
    attack_input_dim=attack_detector.dataset['train'][0].shape[1]
    print("attack_input_dim:",attack_input_dim)               # attack_input_dim: 41
    # attack_detector.def_model(attack_input_dim, output_dim=1, timesteps=1)
    attack_detector.def_model(attack_input_dim, output_dim=1, timesteps=args.timesteps)
    
    return reconnaissance_detector, infection_detector, attack_detector


def init_seq2seq(multistep_dataset, args):
    
    # infection_seq2seq = Seq2seqAttention(name='infection-seq2seq')
    infection_seq2seq = Seq2Seq(name='infection-seq2seq', args=args)
    infection_seq2seq.add_dataset(dataset=multistep_dataset['infection']) 
    print("infection_seq2seq.dataset['train'][0].shape:",infection_seq2seq.dataset['train'][0].shape)
    print("infection_seq2seq.dataset['train'][1].shape:",infection_seq2seq.dataset['train'][1].shape)
    # infection_seq2seq.def_model(XXX)
    # raise Exception("maggie stop")    
    return infection_seq2seq