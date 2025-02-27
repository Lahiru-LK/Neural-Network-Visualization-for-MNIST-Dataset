### 🚀 **Neural Network Visualization for MNIST Dataset** 🧠🔍  

This project is a **Neural Network implementation with visualization** using **Pygame** for the **MNIST handwritten digits dataset**. It provides a **real-time interactive visualization** of the neural network's layers, weights, activations, and predictions.

https://github.com/user-attachments/assets/23659bfd-a624-4845-aeeb-1456cae798f4

---

## **📌 Features**
✅ **Train a Multi-Layer Neural Network** with two hidden layers.  
✅ **Visualize Neurons & Weights** dynamically using **Pygame**.  
✅ **MNIST Dataset** is loaded via **scikit-learn's `fetch_openml`**.  
✅ **Tracks Accuracy per Digit** (e.g., how well the model predicts the number "3").  
✅ **Sparse Weight Visualization** – Shows strong & weak connections in real time.  
✅ **Adjustable Hyperparameters** – Modify **epochs, learning rate, hidden layers**, etc.  

---

## **📂 Project Structure**
```
📦 neural_network_project
 ┣ 📂 .venv/                 # Virtual environment (Python packages)
 ┣ 📂 assets/                # (Optional) For storing images, fonts, etc.
 ┣ 📂 data/                  # (Optional) Dataset storage
 ┣ 📜 .gitignore             # Git ignore file for unnecessary files
 ┣ 📜 data_loader.py         # Loads the MNIST dataset using fetch_openml()
 ┣ 📜 main.py                # Main script to train and run the visualization
 ┣ 📜 neural_network.py      # Neural Network implementation (forward & backpropagation)
 ┣ 📜 requirements.txt       # Dependencies for the project
 ┣ 📜 utils.py               # Utility functions (loss calculation, weight stats, etc.)
 ┣ 📜 visualization.py       # Pygame-based visualization for Neural Network
 ┗ 📜 README.md              # Project documentation (this file)
```

---

## **📥 Installation**
### **🔹 1. Clone the Repository**
```sh
git clone https://github.com/yourusername/neural_network_project.git
cd neural_network_project
```

### **🔹 2. Create a Virtual Environment & Activate it**
```sh
python -m venv .venv
```
- **Windows**:
  ```sh
  .venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```sh
  source .venv/bin/activate
  ```

### **🔹 3. Install Required Dependencies**
```sh
pip install -r requirements.txt
```

### **🔹 4. Run the Project**
```sh
python main.py
```

---

## **📊 MNIST Dataset**
This project uses **MNIST (Modified National Institute of Standards and Technology)** dataset, which consists of **70,000 handwritten digits (0-9)**.

🔹 **Where is the dataset loaded from?**  
📌 The dataset is downloaded **directly from OpenML** using:
```python
from sklearn.datasets import fetch_openml

mnist_data = fetch_openml(name="mnist_784", version=1, as_frame=False)
```
🔹 **How is the data processed?**  
✅ **Normalization:** MNIST images are **28x28 grayscale images**, flattened into **784 pixels**, then normalized to [0,1].  
✅ **One-hot Encoding:** Labels (0-9) are converted to a one-hot encoded format.  

---

## **🖥️ Neural Network Structure**
The model consists of:
- **Input Layer** (784 neurons) – Each neuron represents a pixel.
- **Hidden Layer 1** (1024 neurons, ReLU activation).
- **Hidden Layer 2** (512 neurons, ReLU activation).
- **Output Layer** (10 neurons, Softmax activation for classification).

📌 **Forward Propagation**
```python
self.hidden1_input = np.dot(x, self.weights_input_hidden1) + self.bias_hidden1
self.hidden1_output = np.maximum(0, self.hidden1_input)  # ReLU Activation
self.hidden2_input = np.dot(self.hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
self.hidden2_output = np.maximum(0, self.hidden2_input)  # ReLU Activation
self.output_input = np.dot(self.hidden2_output, self.weights_hidden2_output) + self.bias_output
return self.softmax(self.output_input)
```

📌 **Backpropagation**
```python
output_error = output - y
hidden2_error = np.dot(output_error, self.weights_hidden2_output.T) * (self.hidden2_input > 0)
hidden1_error = np.dot(hidden2_error, self.weights_hidden1_hidden2.T) * (self.hidden1_input > 0)
```

---

## **🎨 Visualization using Pygame**
The neural network's layers, weights, and activations are **visualized in real time**.  
✅ **Connections between neurons** are colored based on weight magnitude.  
✅ **Active neurons glow brighter** depending on activation strength.  
✅ **Side Panel:** Displays **predictions, loss, accuracy, active neurons**, etc.  

🖼 **How to visualize?**  
After training, a **Pygame window opens** showing:
- **Input Layer**
- **Hidden Layers**
- **Output Layer**
- **Prediction & Accuracy Stats**

---

## **🔮 Prediction Process**
1. The model takes a **28x28 handwritten digit image** as input.
2. **Forward propagation** determines the probability of each digit (0-9).
3. The highest probability neuron **determines the predicted digit**.
4. Results are displayed in **Pygame visualization**.

Example **prediction output**:
```plaintext
Epoch 5/30, Test Accuracy: 91.23%
Prediction: 7
Loss: 0.295
Active Neurons: 83.4
```

---

## **⚡ Technologies Used**
| Feature                 | Library        |
|-------------------------|---------------|
| **Machine Learning**    | NumPy, scikit-learn |
| **Visualization**       | Pygame        |
| **Data Processing**     | Pandas (if needed) |
| **Mathematics**        | NumPy (for matrix operations) |
| **Neural Network**      | Custom implementation in Python |

---

## **🌟 Contributing**
Want to improve the project? Follow these steps:

1. **Fork the repository** 🍴
2. **Clone your forked repo**:
   ```sh
   git clone https://github.com/yourusername/neural_network_project.git
   ```
3. **Create a feature branch**:
   ```sh
   git checkout -b feature-name
   ```
4. **Commit your changes**:
   ```sh
   git commit -m "Added new feature"
   ```
5. **Push the branch**:
   ```sh
   git push origin feature-name
   ```
6. **Create a Pull Request** 📩

---

## **📜 License**
This project is licensed under the **MIT License** – feel free to modify and use it!  

---

## **🎯 Future Improvements**
🔹 Improve accuracy using **Dropout & Batch Normalization**.  
🔹 Implement **Convolutional Neural Networks (CNNs)** for better performance.  
🔹 Optimize performance with **GPU acceleration**.  

---


🔥 **If you like this project, don't forget to ⭐ the repo!** 🔥  
🎉 Happy Coding! 🚀👨‍💻

---

### 🎯 **Git Commands to Upload the Project to GitHub**
```sh
git init  # Initialize Git in the project folder
git add .  # Add all files to commit
git commit -m "Initial commit 🚀"
git branch -M main  # Rename branch to main
git remote add origin https://github.com/yourusername/neural_network_project.git  # Add GitHub repo
git push -u origin main  # Push project to GitHub
```
---
