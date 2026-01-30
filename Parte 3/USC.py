import heapq
from collections import defaultdict

# Mapa de costos de las acciones
action_costs = {
    ('Start', 'A'): 5, ('Start', 'B'): 2,
    ('A', 'C'): 4, ('A', 'D'): 2,
    ('B', 'D'): 6, ('B', 'E'): 3,
    ('C', 'Goal'): 8,
    ('D', 'F'): 1, ('D', 'A'): 2,
    ('E', 'F'): 5,
    ('F', 'Goal'): 2
}

# Construcción del grafo no dirigido
graph = defaultdict(list)

for (u, v), cost in action_costs.items():
    graph[u].append((v, cost))
    graph[v].append((u, cost))  # arista inversa

# Implementación de Uniform Cost Search
def uniform_cost_search(start, goal):
    # Cola de prioridad: (costo_acumulado, nodo_actual, camino, traza)
    frontier = [(0, start, [start], [])]
    visited_costs = {}

    while frontier:
        cost, node, path, trace = heapq.heappop(frontier)

        if node == goal:
            return path, trace, cost

        if node in visited_costs and visited_costs[node] <= cost:
            continue

        visited_costs[node] = cost

        for neighbor, action_cost in graph[node]:
            new_cost = cost + action_cost
            new_path = path + [neighbor]

            # Guardar la transición
            new_trace = trace + [
                (node, neighbor, action_cost, new_cost)
            ]

            heapq.heappush(
                frontier,
                (new_cost, neighbor, new_path, new_trace)
            )

    return None, None, float('inf')

# Ejecución del algoritmo
start_node = 'Start'
goal_node = 'Goal'

path, trace, total_cost = uniform_cost_search(start_node, goal_node)

print("Camino de costo mínimo (UCS):")
print(" -> ".join(path))

print("\nDetalle de transiciones:")
for origen, destino, costo_arista, costo_acumulado in trace:
    print(
        f"{origen} -> {destino} | "
        f"costo arista = {costo_arista}, "
        f"costo acumulado = {costo_acumulado}"
    )

print("\nCosto total del camino:", total_cost)
