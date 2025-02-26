import pygame
import numpy as np
from utils import calculate_loss, count_active_neurons, calculate_weight_stats, calculate_digit_accuracy

# Batch Iterator
def batch_iterator(x, y, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]

# Function to visualize the network with metrics and task-specific visualization
def visualize_network_sparse_with_digit_accuracy(
    screen, nn, font, input_data=None, output_data=None, input_image=None, predicted_label=None,
    epoch=None, accuracy=None, loss=None, active_neurons=None, weight_stats=None, digit_accuracy=None, digit=0
):
    # Define layer positions
    input_positions = [(50, 50 + i * 40) for i in range(10)]  # Input layer at x=50
    hidden1_positions = [(350, 50 + i * 40) for i in range(10)]  # Hidden layer 1 at x=350
    hidden2_positions = [(550, 50 + i * 40) for i in range(10)]  # Hidden layer 2 at x=550
    output_positions = [(950, 50 + i * 40) for i in range(10)]  # Output layer at x=950

    # Clear screen with black background
    screen.fill((0, 0, 0))

    # Function to assign color based on weight
    def get_color(weight):
        if weight > 0.5:
            return (0, 255, 81)  # Light Green for strong positive weights
        elif weight < -0.5:
            return (255, 0, 0)  # Red for strong negative weights
        elif abs(weight) > 0.2:
            return (0, 95, 30)  # Dark Green for moderate weights
        elif abs(weight) > 0.1:
            return (0, 49, 107)  # Blue for smaller weights
        else:
            return (0, 35, 18)  # Grey for weak or near-zero weights

    # Function to draw connections between layers
    def draw_thin_connections(layer1_positions, layer2_positions, weights, threshold=0.01, thickness=1):
        for i, (x1, y1) in enumerate(layer1_positions):
            for j, (x2, y2) in enumerate(layer2_positions):
                weight = weights[i, j]
                if abs(weight) > threshold:  # Show connections above threshold
                    color = get_color(weight)
                    pygame.draw.line(screen, color, (x1, y1), (x2, y2), thickness)

    # Set positions of each layer
    input_positions = [(50, 50 + i * 40) for i in range(10)]
    hidden1_positions = [(350, 50 + i * 40) for i in range(10)]
    hidden2_positions = [(550, 50 + i * 40) for i in range(10)]
    output_positions = [(950, 50 + i * 40) for i in range(10)]

    # Draw ultra-thin connections with smaller thresholds (using first 10 neurons from each layer)
    draw_thin_connections(input_positions, hidden1_positions, nn.weights_input_hidden1[:10, :10], threshold=0.005, thickness=1)
    draw_thin_connections(hidden1_positions, hidden2_positions, nn.weights_hidden1_hidden2[:10, :10], threshold=0.005, thickness=1)
    draw_thin_connections(hidden2_positions, output_positions, nn.weights_hidden2_output[:10, :10], threshold=0.005, thickness=1)

    # Function to draw neurons with activation-based intensity
    def draw_neurons(layer_positions, activation_values, color, size=8):
        for i, pos in enumerate(layer_positions):
            if i < len(activation_values[0]):
                intensity = max(80, min(255, int(activation_values[0][i] * 255)))  # Normalize activation intensity
                pygame.draw.circle(screen, (intensity * color[0], intensity * color[1], intensity * color[2]), pos, size)

    # Visualize neurons with layer-specific colors
    if input_data is not None:
        draw_neurons(input_positions, input_data[:, :10], (1, 1, 0), size=10)  # Yellow for input layer

    draw_neurons(hidden1_positions, nn.hidden1_output[:, :10], (0, 0, 1))  # Blue for first hidden layer
    draw_neurons(hidden2_positions, nn.hidden2_output[:, :10], (0, 1, 1))  # Cyan for second hidden layer
    draw_neurons(output_positions, output_data, (1, 0, 1))  # Magenta for output layer

    # Label output neurons
    for i, pos in enumerate(output_positions):
        label = font.render(str(i), True, (255, 255, 255))  # White text
        screen.blit(label, (pos[0] + 20, pos[1] - 10))

    # Draw side panel for additional metrics
    side_panel_x = 1150
    pygame.draw.rect(screen, (50, 50, 50), (side_panel_x, 0, 250, 600))  # Dark grey panel

    # Display input image on the side panel (if provided)
    if input_image is not None:
        input_surface = pygame.surfarray.make_surface(input_image.reshape(28, 28) * 255)
        input_surface = pygame.transform.scale(input_surface, (100, 100))
        screen.blit(input_surface, (side_panel_x + 75, 50))

    # Display metrics on the side panel
    if predicted_label is not None:
        prediction_text = font.render(f"Prediction: {predicted_label}", True, (255, 255, 255))
        screen.blit(prediction_text, (side_panel_x + 20, 180))
    if epoch is not None and accuracy is not None:
        epoch_text = font.render(f"Epoch: {epoch}", True, (255, 255, 255))
        accuracy_text = font.render(f"Accuracy: {accuracy:.4f}", True, (255, 255, 255))
        screen.blit(epoch_text, (side_panel_x + 20, 220))
        screen.blit(accuracy_text, (side_panel_x + 20, 250))
    if loss is not None:
        loss_text = font.render(f"Loss: {loss:.4f}", True, (255, 255, 255))
        screen.blit(loss_text, (side_panel_x + 20, 280))
    if active_neurons is not None:
        active_neurons_text = font.render(f"Active Neurons: {active_neurons:.2f}", True, (255, 255, 255))
        screen.blit(active_neurons_text, (side_panel_x + 20, 310))
    if weight_stats is not None:
        weight_stats_text = font.render(f"Weights (Mean: {weight_stats[0]:.4f}, Std: {weight_stats[1]:.4f})", True, (255, 255, 255))
        screen.blit(weight_stats_text, (side_panel_x + 20, 340))
    if digit_accuracy is not None:
        digit_accuracy_text = font.render(f"Digit {digit} Accuracy: {digit_accuracy:.4f}", True, (255, 255, 255))
        screen.blit(digit_accuracy_text, (side_panel_x + 20, 370))

    # Update display
    pygame.display.flip()


# Train with Visualization and Digit Accuracy
def train_nn_sparse_with_digit_accuracy(nn, x_train, y_train, x_test, y_test, epochs=10, batch_size=64, learning_rate=0.001):
    pygame.init()
    screen = pygame.display.set_mode((1400, 600))  # Adjust width for side panel
    pygame.display.set_caption("Neural Network Sparse Visualization with Digit Accuracy")
    font = pygame.font.Font(None, 24)

    clock = pygame.time.Clock()  # To control frame rate

    digit_to_track = 0  # The specific digit for which accuracy will be displayed

    for epoch in range(epochs):
        for batch_x, batch_y in batch_iterator(x_train, y_train, batch_size):
            # Handle Pygame events to avoid freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Forward pass and backward pass
            outputs = nn.forward(batch_x)
            nn.backward(batch_x, batch_y, outputs, learning_rate)

            # Calculate metrics for the current batch
            loss = calculate_loss(batch_y, outputs)
            accuracy = np.mean(np.argmax(outputs, axis=1) == np.argmax(batch_y, axis=1))  # Batch accuracy
            digit_accuracy = calculate_digit_accuracy(batch_y, outputs, digit_to_track)

            # Visualize a random sample from the batch
            sample_idx = np.random.randint(0, len(batch_x))
            input_data = batch_x[sample_idx].reshape(1, -1)
            output_data = outputs[sample_idx].reshape(1, -1)
            input_image = batch_x[sample_idx]

            visualize_network_sparse_with_digit_accuracy(
                screen, nn, font, input_data, output_data, input_image,
                predicted_label=np.argmax(outputs[sample_idx]),
                epoch=epoch + 1, accuracy=accuracy, loss=loss,
                active_neurons=count_active_neurons(nn.hidden1_output),
                weight_stats=calculate_weight_stats(nn.weights_input_hidden1),
                digit_accuracy=digit_accuracy, digit=digit_to_track
            )

            pygame.display.flip()  # Refresh display
            clock.tick(30)  # Limit frame rate to 30 FPS

        # Evaluate model on the test set after each epoch
        test_outputs = nn.forward(x_test)
        test_accuracy = np.mean(np.argmax(test_outputs, axis=1) == np.argmax(y_test, axis=1))
        print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {test_accuracy:.4f}")

    # Wait for the user to close the window after training is done
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
