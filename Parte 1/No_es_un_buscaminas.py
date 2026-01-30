import random
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, size=5, dirt_prob=0.3, wall_prob=0.2):
        self.size = size
        self.grid = []

        for i in range(size):
            row = []
            for j in range(size):
                r = random.random()
                if r < wall_prob:
                    row.append("WALL")
                elif r < wall_prob + dirt_prob:
                    row.append("DIRT")
                else:
                    row.append("EMPTY")
            self.grid.append(row)

        # Asegurar posición inicial libre
        self.grid[0][0] = "EMPTY"

        # Asegurar al menos una salida desde (0,0)
        if self.grid[0][1] == "WALL" and self.grid[1][0] == "WALL":
            self.grid[0][1] = "EMPTY"

    def is_valid_position(self, x, y):
        return (
            0 <= x < self.size and
            0 <= y < self.size and
            self.grid[x][y] != "WALL"
        )

    def get_cell_state(self, x, y):
        return self.grid[x][y]

    def clean_cell(self, x, y):
        if self.grid[x][y] == "DIRT":
            self.grid[x][y] = "EMPTY"
            return True
        return False


class ReflexAgent:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.cleaned = 0

    def reflex_agent_program(self, percepts):
        position, cell_state, possible_moves = percepts

        # Regla 1: si hay suciedad, limpiar
        if cell_state == "DIRT":
            return "CLEAN"

        # Regla 2 y 3: moverse si es posible
        if possible_moves:
            return random.choice(possible_moves)

        return "IDLE"


# -------- SIMULACIÓN --------

env = Environment(size=5)
agent = ReflexAgent()

moves = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

performance = []

print("Estado inicial del entorno:")
for row in env.grid:
    print(row)

print("\n--- Simulación ---")

for step in range(20):
    cell_state = env.get_cell_state(agent.x, agent.y)

    possible_moves = []
    for action, (dx, dy) in moves.items():
        nx, ny = agent.x + dx, agent.y + dy
        if env.is_valid_position(nx, ny):
            possible_moves.append(action)

    percepts = ((agent.x, agent.y), cell_state, possible_moves)
    action = agent.reflex_agent_program(percepts)

    if action == "CLEAN":
        if env.clean_cell(agent.x, agent.y):
            agent.cleaned += 1

    elif action in moves:
        dx, dy = moves[action]
        agent.x += dx
        agent.y += dy

    performance.append(agent.cleaned)

    print(
        f"Paso {step + 1}: "
        f"Posición=({agent.x},{agent.y}), "
        f"Acción={action}, "
        f"Suciedad limpiada={agent.cleaned}"
    )

# -------- GRÁFICA BONITA 💜 --------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(performance) + 1),
    performance,
    color="purple",
    marker="o",
    linewidth=2,
    markersize=6
)

plt.xlabel("Tiempo (pasos)")
plt.ylabel("Cantidad de suciedad limpiada")
plt.title("Rendimiento del Agente Reactivo", fontsize=13)

# Vértices bien definidos
plt.xticks(range(1, 21))
plt.yticks(range(0, max(performance) + 2))

plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
