import torch
import torch.nn.functional as F

# FGSM attack code
def fgsm_attack(model, data, target, epsilon):
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)
    init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    # Calculate the loss
    loss = F.nll_loss(output, target.view(output.shape[0]))

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_data = data.clone().detach() + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    data.requires_grad = False

    # Return the perturbed image
    return perturbed_data

# PGD attack code
def pgd_attack(model, data, target, epsilon, num_steps=7, alpha=0.01, random_start=True,
        d_min=0.0, d_max=1.0):

    original_data = data.clone()
    perturbed_data = data
    perturbed_data.requires_grad = True

    data_max = data + epsilon
    data_min = data - epsilon

    # temporarily no clamp for tabular data
    data_max.clamp_(d_min, d_max)
    data_min.clamp_(d_min, d_max)

    if random_start:
        with torch.no_grad():
            perturbed_data.data = original_data + perturbed_data.uniform_(-1*epsilon, epsilon)
            perturbed_data.data.clamp_(d_min, d_max)

    for _ in range(num_steps):
        output = model(perturbed_data)

        loss = F.cross_entropy(output, target.view(output.shape[0]))

        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        loss.backward()
        data_grad = perturbed_data.grad

        with torch.no_grad():
            perturbed_data.data += epsilon * torch.sign(data_grad)
            perturbed_data.data = torch.max(torch.min(perturbed_data, data_max),
                                                      data_min)

    perturbed_data.requires_grad = False

    return perturbed_data

def adversarial_eval(model, device, eval_loader, epsilon):
    # Accuracy counter
    fgsm_correct = 0
    pgd_correct = 0

    # Store adversarial examples
    fgsm_example = []
    pgd_example = []

    # Loop over all examples in test set
    for i, (data, target) in enumerate(eval_loader):
        if i == 1000:
            break

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        fgsm_data = data.clone()
        fgsm_target = target.clone()
        fgsm_data, fgsm_target = fgsm_data.to(device), fgsm_target.to(device)

        pgd_data = data.clone()
        pgd_target = target.clone()
        pgd_data, pgd_target = pgd_data.to(device), pgd_target.to(device)

        # Forward pass the data through the model
        output = model(data)

        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Call FGSM Attack
        fgsm_perturbed_data = fgsm_attack(model, fgsm_data, fgsm_target, epsilon)

        # Call PGD Attack
        pgd_perturbed_data = pgd_attack(model, pgd_data, pgd_target, epsilon)

        # Re-classify the perturbed image
        fgsm_output = model(fgsm_perturbed_data)
        pgd_output = model(pgd_perturbed_data)

        # Check for success
        fgsm_final_pred = fgsm_output.max(1, keepdim=True)[1] # get the index of the max log-probability
        pgd_final_pred = pgd_output.max(1, keepdim=True)[1]

        if fgsm_final_pred.item() == target.item():
            fgsm_correct += 1
        else:
            fgsm_example.append((fgsm_perturbed_data, target))

        if pgd_final_pred.item() == target.item():
            pgd_correct += 1
        else:
            pgd_example.append((pgd_perturbed_data, target))

    # Calculate final accuracy for this epsilon
    fgsm_final_acc = round(fgsm_correct / float(1000), 4)
    pgd_final_acc = round(pgd_correct / float(1000), 4)

    # Return the accuracy and an adversarial example
    return fgsm_final_acc, pgd_final_acc, fgsm_example, pgd_example
