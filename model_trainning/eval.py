import matplotlib.pyplot as plt
import torch


def unNormalizeSteering(linear, angular):

    # mean = [1.84769514e-01 ,3.24062102e-03]
    # std  = [0.06764629 ,0.41442807]

    mean = [1.82032557e-01, 2.72741526e-03]
    std = [0.07903122, 0.41434146]

    linear = (linear * std[0]) + mean[0]
    angular = (angular * std[1]) + mean[1]

    return linear, angular


def plotVelocities(evalDataLoader, device, model,root_dir, model_arch="cnn"):
    index = []
    linear_pred_list = []
    linear_actl_list = []
    angular_pred_list = []
    angular_actl_list = []
    prev_features = None
    first_iter = True
    # propagate the eval dataset
    with torch.no_grad():
        for i, samples in enumerate(evalDataLoader):
            if i == 2008:
                break
            scan, goal, steering = samples["scan"].to(device, dtype=torch.float), samples["goal"].to(
                device, dtype=torch.float), samples["steering"].to(device, dtype=torch.float)

            if model_arch == "cnn":
                output = model(scan, goal)
            else:

                output, features = model(scan, goal, prev_features)
                if first_iter:
                  prev_features = features.repeat(1,8,1)
                  first_iter = False
                  continue
                else:
                  prev_features = features

            linear_actl, angular_actl = unNormalizeSteering(
                steering[0][0].item(), steering[0][1].item())
            linear_pred, angular_pred = unNormalizeSteering(
                output[0].item(), output[1].item())
            index.append(i)
            linear_pred_list.append(linear_pred)
            linear_actl_list.append(linear_actl)
            angular_pred_list.append(angular_pred)
            angular_actl_list.append(angular_actl)
    # plot and compare the predicted and actual velocities
    fig = plt.figure(num=None, figsize=(24, 9), dpi=150.0)  # create the canvas for plotting
    linear = plt.subplot(2, 1, 2)
    # (2,1,1) indicates total number of rows, columns, and figure number respectively
    angular = plt.subplot(2, 1, 1)

    #fig.suptitle('predicted vs actual steering commands',fontsize=30.0)

    linear.plot(index, linear_actl_list, 'g', label="actual")
    linear.plot(index, linear_pred_list, 'y', label="predicted")
    linear.set_ylabel('    v [m/s]',fontsize=20.0)
    linear.set_xlabel('frame',fontsize=25.0)

    angular.plot(index, angular_actl_list, 'g')
    angular.plot(index, angular_pred_list, 'y')
    angular.set_ylabel('    ω [rad/s]',fontsize=20.0)
    # angular.set_xlabel('frame')
    fig.legend(fancybox=True, framealpha=1, shadow=True, fontsize=18.0,loc='upper right', bbox_to_anchor=(0.91, 0.91))
    fig.savefig(root_dir+"images/steering_comands_compar_"+model_arch+".png")
