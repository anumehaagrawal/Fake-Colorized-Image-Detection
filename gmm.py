from histogram import hue_sat_calculation
import pandas as pd
from sklearn.mixture import GaussianMixture

def gmm():
    hue_fake,hue_true,sat_fake,sat_true,dc_fake,dc_true,bc_fake, bc_true  = hue_sat_calculation()
    training_set = []
    for img in range(len(hue_fake)):
        image_data = []
        for i in range(len(hue_fake[img])):

            image_data.append(hue_fake[img][i][0])
            image_data.append(sat_fake[img][i][0])
            image_data.append(dc_fake[img][i][0])
            image_data.append(bc_fake[img][i][0])

        
        training_set.append(image_data)
        image_data = []
        for i in range(len(hue_true[img])):
            image_data.append(hue_true[img][i][0])
            image_data.append(sat_true[img][i][0])
            image_data.append(dc_true[img][i][0])
            image_data.append(bc_true[img][i][0])
    

        training_set.append(image_data)
    df = pd.DataFrame(training_set)
    GMM = GaussianMixture(n_components=3).fit(df) # Instantiate and fit the model
    print('Converged:',GMM.converged_) # Check if the model has converged
    means = GMM.means_ 
    covariances = GMM.covariances_
    print("mean", means)
    print("cv",len(covariances[0][0]))
    

gmm()