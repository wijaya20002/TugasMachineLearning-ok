import numpy as np

# Definisi environment (maze 3x3)
maze = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 2]
])

# Inisialisasi Q-table
q_table = np.zeros((9, 4))

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1

def take_action(state, action, maze):
    row = state // 3
    col = state % 3

    if action == 0:
        row -= 1
    elif action == 1:
        row += 1
    elif action == 2:
        col -= 1
    elif action == 3:
        col += 1

    if row < 0 or row >= 3 or col < 0 or col >= 3:
        return state, -1, False

    if maze[row][col] == 1:
        return state, -1, False

    if maze[row][col] == 2:
        next_state = row * 3 + col
        return next_state, 10, True

    next_state = row * 3 + col
    return next_state, -0.1, False


# ================= TRAINING =================
for episode in range(1000):
    state = 0
    done = False

    while not done:
        # Pilih aksi
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, 4)
        else:
            action = np.argmax(q_table[state])

        # Ambil langkah
        next_state, reward, done = take_action(state, action, maze)

        # Update Q-table
        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state


# ================= OUTPUT =================
print("Q-table hasil training:")
print(q_table)


# ================= TEST JALUR =================
state = 0
done = False
path = [state]

while not done:
    action = np.argmax(q_table[state])
    state, _, done = take_action(state, action, maze)
    path.append(state)

print("Jalur terbaik:", path)