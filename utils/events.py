import numpy as np
from utils.softmax import softmax

def get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_train_windows_x):
    
    events = []
    for detector in [reconnaissance_detector, infection_detector, attack_detector]:
        malicious_class_score = detector.model.predict(cle_train_windows_x)  
        # print("malicious_class_score.shape:",malicious_class_score.shape)      
        # malicious_class_score.shape: (19808, 1)
        
        malicious_events_proportion = np.sum(np.array(malicious_class_score)>0.5)/len(malicious_class_score)
        print(f'proportion of malicious events tagged by {detector.modelname} is: { 100*malicious_events_proportion:.4f} %')
        
        events.append(malicious_class_score)

    # print("events_num:",events[0].shape[0])   
    """ 
    proportion of malicious events tagged by reconnaissance-detector is: 0.48137116316639744
    proportion of malicious events tagged by infection-detector is: 0.4333602584814216
    proportion of malicious events tagged by attack-detector is: 0.17200121163166399
    events_num: 19808
    """
    
    benign_class_score = []    
    for i in range(events[0].shape[0]):
        reconnaissance_prob = events[0][i]
        infection_prob = events[1][i]
        attack_prob = events[2][i]
        
        not_reconnaissance_prob = 1-events[0][i]
        not_infection_prob = 1-events[1][i]
        not_attack_prob = 1-events[2][i]
        
        benign_prob = not_reconnaissance_prob * not_infection_prob * not_attack_prob
        benign_class_score.append(benign_prob)    
        
        # print("i:",i)
        # print("reconnaissance_prob:",reconnaissance_prob)
        # print("not_reconnaissance_prob:",not_reconnaissance_prob)
        # print("infection_prob:",infection_prob)
        # print("not_infection_prob:",not_infection_prob)
        # print("attack_prob:",attack_prob)
        # print("not_attack_prob:",not_attack_prob)
        # print("benign_prob:",benign_prob)
        """ 
        i: 19807
        reconnaissance_prob: [0.03644627]
        not_reconnaissance_prob: [0.9635537]
        infection_prob: [0.9949181]
        not_infection_prob: [0.00508189]
        attack_prob: [1.6030718e-08]
        not_attack_prob: [1.]
        benign_prob: [0.00489668]      
        """
        
    benign_events_proportion = np.sum(np.array(benign_class_score)>0.5)/len(benign_class_score)
    print(f'proportion of events estimated to be benign is: { benign_events_proportion}')
    events.append(benign_class_score)

    """ 
    proportion of events estimated to be benign is: 0.038115912762520195
    """ 
       
    events=np.array(events).squeeze()
    # print("events.shape:",events.shape)
    """ 
    events.shape: (4, 19808)
    """
    events = np.transpose(events)
    # print("events.shape:",events.shape)
    # print("events[:1]:",events[:1])
    """ 
    events.shape: (19808, 4)
    events[:1]: [[0.11222175 0.99411726 0.0520148  0.00495092]]
    """    
    
    events = [softmax(e) for e in events]
    events=np.array(events)
    # print("events.shape:",events.shape)
    # print("events[:1]:",events[:1])   
    
    """ 
    events.shape: (19808, 4)
    events[:1]: [[0.19028315 0.45962402 0.17916484 0.17092799]]
    """
    return events 