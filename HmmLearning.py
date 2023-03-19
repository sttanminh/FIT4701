from hmmlearn import hmm
import numpy as np


#This LDA is from 3D pose from stack overflow, beside sklear.discriminant_anaylysis can be used 

def compute_joint_angles(joint_positions):
    # Compute joint angles from joint positions
    num_joints = joint_positions.shape[0]
    joint_angles = np.zeros((num_joints-1, 3))
    for i in range(1, num_joints):
        parent_joint = joint_positions[i-1]
        joint = joint_positions[i]
        # Compute the joint angle relative to the parent joint
        # using some mathematical formula or convention
        angle = compute_angle(parent_joint, joint)
        joint_angles[i-1] = angle
    return joint_angles


def extract_feature(poses):
    # Compute joint angles for each frame in the pose sequence
    angles = []
    for pose in poses:
        joint_angles = compute_joint_angles(pose)
        angles.append(joint_angles)
    angles = np.array(angles)

    # Flatten the angles into a feature vector
    n_frames, n_joints, n_angles = angles.shape
    features = angles.reshape((n_frames, n_joints * n_angles))

    return features





def hmmTraining():

    #Define paremeter for HMM training ( hidden states, startProb , transmat, emissionprb, covariance_type ,n_iter )
    #3D pose data (frames,19,3), training and test data 
    #Load data
    TestData = []
    LabelData = []
    labels = [1,2,3,4]


    #LDA (feature extract) 
    features = extract_feature(LabelData)
    testDataExtracted = extract_feature(TestData)



    #Data quantization
    n_clusters = 16  # depend on data complexity and computational time expectation
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)
    quantized_features = kmeans.predict(features)
    test_quantized_features = kmeans.predict(testDataExtracted)




    #Training hmm on feature vector
    n_components = 3  # choose the number of hidden states
    model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag')
    model.fit(quantized_features)



    #Recognizing data using hmm
    probs = []
    for i in range(len(test_quantized_features)):
        score = model.score(test_quantized_features[i])
        probs.append(score)
    activity_index = np.argmax(probs)
    activity_label = labels[activity_index]

    return activity_label

