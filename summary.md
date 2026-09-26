# Fashion-MNIST Image Classifier: Development Summary

## Objective
Build an image classifier in PyTorch that sorts Fashion-MNIST images into 10 clothing categories. The work was split into five scripts, each building on the previous one.

## 1. Dataset preparation (`01_dataset.py`)
- Loaded Fashion-MNIST through TorchVision: 70,000 grayscale 28×28 images in 10 classes.
- Converted the images to float32 tensors with pixel values scaled to [0, 1] using `transforms.ToTensor()`.
- Split the original 60,000 training images into **55,000 for training and 5,000 for validation**, keeping the **10,000 test images** separate.
- Used **random seed 42** so the split and the training shuffle are the same on every run.
- Created DataLoaders with a **batch size of 32**. Only the training loader shuffles its data.

## 2. Model architecture (`02_model.py`)
The model is a multilayer perceptron built with `nn.Module`:

| Layer | Output size | Parameters |
|---|---|---|
| Flatten (28×28) | 784 | 0 |
| Dense + ReLU | 300 | 235,500 |
| Dense + ReLU | 100 | 30,100 |
| Dense (output) | 10 | 1,010 |
| **Total** | | **266,610** |

- The model runs on the GPU if one is available and on the CPU otherwise.
- It outputs raw scores (logits), which is the input `CrossEntropyLoss` expects.

## 3. Training (`03_training.py`)
- Loss function: `CrossEntropyLoss`. Optimizer: **SGD with a learning rate of 0.1**. Trained for **5 epochs**.
- Training loss, training accuracy, validation loss and validation accuracy were recorded after every epoch.
- The trained weights and the metrics were saved to files for the later scripts.

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---|---|---|---|---|
| 1 | 0.6045 | 77.83% | 0.4279 | 84.12% |
| 2 | 0.4080 | 84.93% | 0.4050 | 84.96% |
| 3 | 0.3626 | 86.39% | 0.4043 | 85.46% |
| 4 | 0.3355 | 87.47% | 0.3647 | 86.76% |
| 5 | 0.3149 | 88.19% | 0.3446 | 87.20% |

## 4. Visualization (`04_accuracy_plot.py`)
- A Matplotlib line graph shows training and validation accuracy per epoch, with labeled axes, a legend and a title.
- Training accuracy rose from 77.8% to 88.2% and validation accuracy from 84.1% to 87.2%.
- The final gap between the two is only about 1 percentage point.

## 5. Evaluation and predictions (`05_evaluation.py`)
- The trained model was switched to evaluation mode with `model.eval()`, and gradients were turned off with `torch.no_grad()`.
- Three validation images were classified. Softmax turned the model's scores into probabilities, and the four most likely categories were shown for each image:

| Image | Actual | Predicted | Top probability | Runner-up |
|---|---|---|---|---|
| 1 | Sneaker | Sneaker ✓ | 93.14% | Ankle boot 4.61% |
| 2 | Coat | Coat ✓ | 93.75% | Pullover 6.05% |
| 3 | Pullover | Pullover ✓ | 90.10% | Coat 5.61% |

- Results were moved back to the CPU before they were printed.
- **Test accuracy: 87.13%** on the 10,000 test images.

## Conclusion
The simple two-hidden-layer network reached **87.13% accuracy on the test set** after only 5 epochs of training with plain SGD.

- **The model generalizes well.** Test accuracy (87.13%) is almost the same as the final validation accuracy (87.20%), so the validation set gave a reliable estimate of performance on new images.
- **There is no sign of overfitting yet.** Training and validation accuracy differ by only about 1 point, and both losses were still going down at epoch 5.
- **The mistakes are sensible.** When the model is unsure, the runner-up is a visually similar item: sneaker vs. ankle boot, coat vs. pullover.

**Limitations and possible improvements:**
- **Flattening loses image structure.** The network treats each image as a flat list of 784 pixels, so it can't use how neighboring pixels relate. A convolutional neural network (CNN) usually reaches over 90% on Fashion-MNIST.
- **Training was short.** More epochs, a learning-rate schedule, or an optimizer such as Adam would likely raise accuracy.
- **Regularization isn't needed yet but will be.** With longer training, dropout or early stopping would help prevent overfitting.

Overall, the project covered the full deep learning workflow: preparing data, defining a model, training it, plotting the results and evaluating it.

## How to run
```bash
pip install torch torchvision numpy matplotlib
python 01_dataset.py        # download data and check the splits
python 02_model.py          # print the model and check output shapes
python 03_training.py       # train for 5 epochs and save weights + history
python 04_accuracy_plot.py  # plot training vs. validation accuracy
python 05_evaluation.py     # sample predictions and test accuracy
```
