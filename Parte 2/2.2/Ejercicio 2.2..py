"""
Ejercicio 2.2: Breadth-First Search (BFS) vs Depth-First Search (DFS)
Implementación desde cero con visualización detallada
"""

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Dict, Tuple, Optional, Set
import os

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

OUTPUT_DIR = os.path.join(os.getcwd(), 'ejercicio_2_2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("EJERCICIO 2.2: BFS vs DFS - ANÁLISIS COMPARATIVO")
print("=" * 70)
print(f"📁 Guardando resultados en: {OUTPUT_DIR}\n")

# Mapa de la bodega
warehouse_map = {
    'Start': ['A', 'B'],
    'A': ['Start', 'C', 'D'],
    'B': ['Start', 'D', 'E'],
    'C': ['A', 'Goal'],
    'D': ['A', 'B', 'F'],
    'E': ['B', 'F'],
    'F': ['D', 'E', 'Goal'],
    'Goal': ['C', 'F']
}


# ============================================================================
# CREACIÓN DEL GRAFO
# ============================================================================

def create_graph(adjacency_dict: Dict[str, List[str]]) -> nx.Graph:
    """
    Crea un grafo de NetworkX desde el diccionario de adyacencias.
    """
    G = nx.Graph()
    for node, neighbors in adjacency_dict.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    return G


# ============================================================================
# BREADTH-FIRST SEARCH (BFS) - IMPLEMENTACIÓN DESDE CERO
# ============================================================================

def bfs_from_scratch(graph: nx.Graph, start: str, goal: str) -> Tuple[Optional[List[str]], Set[str], List[str]]:
    """
    Implementación de BFS desde cero usando cola (FIFO).

    Args:
        graph: Grafo de NetworkX
        start: Nodo inicial
        goal: Nodo objetivo

    Returns:
        Tupla con:
        - Camino encontrado (lista de nodos)
        - Nodos visitados (conjunto)
        - Orden de exploración (lista)
    """
    # Cola FIFO: almacena tuplas (nodo_actual, camino_hasta_ese_nodo)
    queue = deque([(start, [start])])

    # Conjunto de nodos visitados
    visited = set([start])

    # Lista que mantiene el orden de exploración
    exploration_order = [start]

    print("🔵 Ejecutando BFS (Breadth-First Search)...")
    print(f"   Estructura de datos: Cola FIFO (First In, First Out)")
    print(f"   Estrategia: Exploración por niveles\n")

    iteration = 0

    while queue:
        # FIFO: Sacar el primer elemento de la cola
        current_node, path = queue.popleft()

        iteration += 1
        print(f"   Iteración {iteration}: Explorando '{current_node}' | Cola: {[n for n, _ in queue]}")

        # ¿Llegamos al objetivo?
        if current_node == goal:
            print(f"\n   ✅ ¡Meta alcanzada en iteración {iteration}!")
            print(f"   📍 Camino encontrado: {' → '.join(path)}")
            print(f"   📊 Nodos visitados: {len(visited)}")
            return path, visited, exploration_order

        # Explorar vecinos (en orden alfabético para consistencia)
        neighbors = sorted(graph.neighbors(current_node))

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                exploration_order.append(neighbor)
                # Agregar al final de la cola con el camino extendido
                queue.append((neighbor, path + [neighbor]))
                print(f"      → Agregando '{neighbor}' a la cola")

    # No se encontró camino
    print(f"\n   ❌ No se encontró camino al objetivo")
    return None, visited, exploration_order


# ============================================================================
# DEPTH-FIRST SEARCH (DFS) - IMPLEMENTACIÓN DESDE CERO
# ============================================================================

def dfs_from_scratch(graph: nx.Graph, start: str, goal: str) -> Tuple[Optional[List[str]], Set[str], List[str]]:
    """
    Implementación de DFS desde cero usando pila (LIFO).

    Args:
        graph: Grafo de NetworkX
        start: Nodo inicial
        goal: Nodo objetivo

    Returns:
        Tupla con:
        - Camino encontrado (lista de nodos)
        - Nodos visitados (conjunto)
        - Orden de exploración (lista)
    """
    # Pila LIFO: almacena tuplas (nodo_actual, camino_hasta_ese_nodo)
    stack = [(start, [start])]

    # Conjunto de nodos visitados
    visited = set()

    # Lista que mantiene el orden de exploración
    exploration_order = []

    print("\n🔴 Ejecutando DFS (Depth-First Search)...")
    print(f"   Estructura de datos: Pila LIFO (Last In, First Out)")
    print(f"   Estrategia: Exploración en profundidad\n")

    iteration = 0

    while stack:
        # LIFO: Sacar el último elemento de la pila
        current_node, path = stack.pop()

        # Si ya visitamos este nodo, saltar
        if current_node in visited:
            continue

        # Marcar como visitado
        visited.add(current_node)
        exploration_order.append(current_node)

        iteration += 1
        print(f"   Iteración {iteration}: Explorando '{current_node}' | Pila: {[n for n, _ in stack]}")

        # ¿Llegamos al objetivo?
        if current_node == goal:
            print(f"\n   ✅ ¡Meta alcanzada en iteración {iteration}!")
            print(f"   📍 Camino encontrado: {' → '.join(path)}")
            print(f"   📊 Nodos visitados: {len(visited)}")
            return path, visited, exploration_order

        # Explorar vecinos (en orden inverso para mantener orden alfabético en la exploración)
        neighbors = sorted(graph.neighbors(current_node), reverse=True)

        for neighbor in neighbors:
            if neighbor not in visited:
                # Agregar al tope de la pila con el camino extendido
                stack.append((neighbor, path + [neighbor]))
                print(f"      → Agregando '{neighbor}' a la pila")

    # No se encontró camino
    print(f"\n   ❌ No se encontró camino al objetivo")
    return None, visited, exploration_order


# ============================================================================
# VISUALIZACIÓN MEJORADA
# ============================================================================

def visualize_search_algorithm(graph: nx.Graph,
                               path: Optional[List[str]],
                               visited: Set[str],
                               exploration_order: List[str],
                               algorithm_name: str,
                               filename: str):
    """
    Visualiza el grafo resaltando los nodos visitados y el camino encontrado.

    Args:
        graph: Grafo de NetworkX
        path: Camino encontrado
        visited: Conjunto de nodos visitados
        exploration_order: Orden en que se exploraron los nodos
        algorithm_name: Nombre del algoritmo (BFS o DFS)
        filename: Nombre del archivo de salida
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Layout consistente para ambos grafos
    pos = nx.spring_layout(graph, seed=42, k=2, iterations=50)

    # ========== PANEL IZQUIERDO: Nodos visitados ==========
    ax1.set_title(f'{algorithm_name}: Nodos Visitados', fontsize=16, fontweight='bold')

    # Colorear nodos según fueron visitados
    node_colors = []
    node_sizes = []

    for node in graph.nodes():
        if node == 'Start':
            node_colors.append('lightgreen')
            node_sizes.append(3000)
        elif node == 'Goal':
            node_colors.append('lightcoral')
            node_sizes.append(3000)
        elif node in visited:
            node_colors.append('yellow')
            node_sizes.append(2500)
        else:
            node_colors.append('lightgray')
            node_sizes.append(2000)

    # Dibujar nodos
    nx.draw_networkx_nodes(graph, pos, ax=ax1, node_color=node_colors,
                           node_size=node_sizes, alpha=0.9, edgecolors='black', linewidths=2)

    # Dibujar arcos
    nx.draw_networkx_edges(graph, pos, ax=ax1, width=2, alpha=0.3, edge_color='gray')

    # Dibujar etiquetas de nodos
    nx.draw_networkx_labels(graph, pos, ax=ax1, font_size=12, font_weight='bold')

    # Agregar números de orden de exploración
    exploration_labels = {node: f"({exploration_order.index(node) + 1})"
                          for node in exploration_order}
    label_pos = {k: (v[0], v[1] - 0.15) for k, v in pos.items()}
    nx.draw_networkx_labels(graph, label_pos, exploration_labels, ax=ax1,
                            font_size=9, font_color='blue')

    # Leyenda
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                   markersize=12, label='Inicio', markeredgecolor='black', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral',
                   markersize=12, label='Meta', markeredgecolor='black', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                   markersize=12, label='Visitado', markeredgecolor='black', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                   markersize=12, label='No visitado', markeredgecolor='black', markeredgewidth=2)
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)
    ax1.axis('off')

    # ========== PANEL DERECHO: Camino encontrado ==========
    ax2.set_title(f'{algorithm_name}: Camino Encontrado', fontsize=16, fontweight='bold')

    # Colorear nodos del camino
    path_node_colors = []
    path_node_sizes = []

    for node in graph.nodes():
        if node == 'Start':
            path_node_colors.append('lightgreen')
            path_node_sizes.append(3000)
        elif node == 'Goal':
            path_node_colors.append('lightcoral')
            path_node_sizes.append(3000)
        elif path and node in path:
            path_node_colors.append('gold')
            path_node_sizes.append(2500)
        else:
            path_node_colors.append('lightgray')
            path_node_sizes.append(2000)

    # Dibujar nodos
    nx.draw_networkx_nodes(graph, pos, ax=ax2, node_color=path_node_colors,
                           node_size=path_node_sizes, alpha=0.9, edgecolors='black', linewidths=2)

    # Dibujar arcos normales
    nx.draw_networkx_edges(graph, pos, ax=ax2, width=2, alpha=0.3, edge_color='gray')

    # Resaltar el camino encontrado
    if path and len(path) > 1:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(graph, pos, ax=ax2, edgelist=path_edges,
                               width=5, alpha=0.9, edge_color='red',
                               arrows=True, arrowsize=20, arrowstyle='->')

    # Dibujar etiquetas
    nx.draw_networkx_labels(graph, pos, ax=ax2, font_size=12, font_weight='bold')

    # Información del camino
    if path:
        info_text = f"Camino: {' → '.join(path)}\n"
        info_text += f"Longitud: {len(path)} nodos ({len(path) - 1} saltos)\n"
        info_text += f"Nodos visitados: {len(visited)}/{graph.number_of_nodes()}"

        ax2.text(0.5, -0.15, info_text, transform=ax2.transAxes,
                 fontsize=11, ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                           edgecolor='orange', linewidth=2))

    # Leyenda
    path_legend = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                   markersize=12, label='En el camino', markeredgecolor='black', markeredgewidth=2),
        plt.Line2D([0], [0], color='red', linewidth=4, label='Camino óptimo')
    ]
    ax2.legend(handles=path_legend, loc='upper left', fontsize=10)
    ax2.axis('off')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n   ✅ Visualización guardada: {filename}")
    plt.close()


# ============================================================================
# COMPARACIÓN LADO A LADO
# ============================================================================

def visualize_comparison(graph: nx.Graph,
                         bfs_path: List[str], bfs_visited: Set[str],
                         dfs_path: List[str], dfs_visited: Set[str]):
    """
    Crea una comparación lado a lado de BFS y DFS.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    pos = nx.spring_layout(graph, seed=42, k=2, iterations=50)

    # ========== BFS (Izquierda) ==========
    ax1.set_title('BFS: Búsqueda en Anchura', fontsize=16, fontweight='bold', color='blue')

    bfs_colors = []
    for node in graph.nodes():
        if node == 'Start':
            bfs_colors.append('lightgreen')
        elif node == 'Goal':
            bfs_colors.append('lightcoral')
        elif bfs_path and node in bfs_path:
            bfs_colors.append('deepskyblue')
        elif node in bfs_visited:
            bfs_colors.append('lightyellow')
        else:
            bfs_colors.append('lightgray')

    nx.draw_networkx_nodes(graph, pos, ax=ax1, node_color=bfs_colors,
                           node_size=2500, alpha=0.9, edgecolors='black', linewidths=2)
    nx.draw_networkx_edges(graph, pos, ax=ax1, width=2, alpha=0.3, edge_color='gray')

    if bfs_path and len(bfs_path) > 1:
        bfs_edges = [(bfs_path[i], bfs_path[i + 1]) for i in range(len(bfs_path) - 1)]
        nx.draw_networkx_edges(graph, pos, ax=ax1, edgelist=bfs_edges,
                               width=4, alpha=0.9, edge_color='blue',
                               arrows=True, arrowsize=15)

    nx.draw_networkx_labels(graph, pos, ax=ax1, font_size=12, font_weight='bold')

    # Info BFS
    bfs_info = f"Camino: {' → '.join(bfs_path)}\n"
    bfs_info += f"Saltos: {len(bfs_path) - 1} | Visitados: {len(bfs_visited)}"
    ax1.text(0.5, -0.1, bfs_info, transform=ax1.transAxes,
             fontsize=11, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                       edgecolor='blue', linewidth=2))
    ax1.axis('off')

    # ========== DFS (Derecha) ==========
    ax2.set_title('DFS: Búsqueda en Profundidad', fontsize=16, fontweight='bold', color='red')

    dfs_colors = []
    for node in graph.nodes():
        if node == 'Start':
            dfs_colors.append('lightgreen')
        elif node == 'Goal':
            dfs_colors.append('lightcoral')
        elif dfs_path and node in dfs_path:
            dfs_colors.append('salmon')
        elif node in dfs_visited:
            dfs_colors.append('lightyellow')
        else:
            dfs_colors.append('lightgray')

    nx.draw_networkx_nodes(graph, pos, ax=ax2, node_color=dfs_colors,
                           node_size=2500, alpha=0.9, edgecolors='black', linewidths=2)
    nx.draw_networkx_edges(graph, pos, ax=ax2, width=2, alpha=0.3, edge_color='gray')

    if dfs_path and len(dfs_path) > 1:
        dfs_edges = [(dfs_path[i], dfs_path[i + 1]) for i in range(len(dfs_path) - 1)]
        nx.draw_networkx_edges(graph, pos, ax=ax2, edgelist=dfs_edges,
                               width=4, alpha=0.9, edge_color='red',
                               arrows=True, arrowsize=15)

    nx.draw_networkx_labels(graph, pos, ax=ax2, font_size=12, font_weight='bold')

    # Info DFS
    dfs_info = f"Camino: {' → '.join(dfs_path)}\n"
    dfs_info += f"Saltos: {len(dfs_path) - 1} | Visitados: {len(dfs_visited)}"
    ax2.text(0.5, -0.1, dfs_info, transform=ax2.transAxes,
             fontsize=11, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='mistyrose',
                       edgecolor='red', linewidth=2))
    ax2.axis('off')

    # Título general
    fig.suptitle('Comparación: BFS vs DFS', fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '04_comparacion_bfs_dfs.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n   ✅ Comparación guardada: 04_comparacion_bfs_dfs.png")
    plt.close()


# ============================================================================
# ANÁLISIS Y REPORTE
# ============================================================================

def create_analysis_report(graph: nx.Graph,
                           bfs_path: List[str], bfs_visited: Set[str], bfs_order: List[str],
                           dfs_path: List[str], dfs_visited: Set[str], dfs_order: List[str]):
    """
    Crea un reporte de análisis detallado.
    """
    report_path = os.path.join(OUTPUT_DIR, 'ANALISIS_BFS_DFS.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("EJERCICIO 2.2: ANÁLISIS COMPARATIVO BFS vs DFS\n")
        f.write("=" * 70 + "\n\n")

        # Información del grafo
        f.write("1. INFORMACIÓN DEL GRAFO\n")
        f.write("-" * 70 + "\n")
        f.write(f"Nodos totales: {graph.number_of_nodes()}\n")
        f.write(f"Arcos totales: {graph.number_of_edges()}\n")
        f.write(f"Nodos: {sorted(graph.nodes())}\n\n")

        # Resultados BFS
        f.write("2. BREADTH-FIRST SEARCH (BFS)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Camino encontrado: {' → '.join(bfs_path)}\n")
        f.write(f"Número de saltos (hops): {len(bfs_path) - 1}\n")
        f.write(f"Nodos en el camino: {len(bfs_path)}\n")
        f.write(f"Nodos visitados: {len(bfs_visited)}\n")
        f.write(f"Orden de exploración: {' → '.join(bfs_order)}\n")
        f.write(f"Porcentaje de nodos explorados: {len(bfs_visited) / graph.number_of_nodes() * 100:.1f}%\n\n")

        # Resultados DFS
        f.write("3. DEPTH-FIRST SEARCH (DFS)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Camino encontrado: {' → '.join(dfs_path)}\n")
        f.write(f"Número de saltos (hops): {len(dfs_path) - 1}\n")
        f.write(f"Nodos en el camino: {len(dfs_path)}\n")
        f.write(f"Nodos visitados: {len(dfs_visited)}\n")
        f.write(f"Orden de exploración: {' → '.join(dfs_order)}\n")
        f.write(f"Porcentaje de nodos explorados: {len(dfs_visited) / graph.number_of_nodes() * 100:.1f}%\n\n")

        # Comparación
        f.write("4. COMPARACIÓN\n")
        f.write("-" * 70 + "\n")
        f.write(f"BFS - Saltos: {len(bfs_path) - 1} | Visitados: {len(bfs_visited)}\n")
        f.write(f"DFS - Saltos: {len(dfs_path) - 1} | Visitados: {len(dfs_visited)}\n\n")

        # Determinar ganador
        if len(bfs_path) < len(dfs_path):
            winner = "BFS"
            difference = len(dfs_path) - len(bfs_path)
            f.write(f"🏆 GANADOR: BFS encontró un camino {difference} salto(s) más corto\n\n")
        elif len(dfs_path) < len(bfs_path):
            winner = "DFS"
            difference = len(bfs_path) - len(dfs_path)
            f.write(f"🏆 GANADOR: DFS encontró un camino {difference} salto(s) más corto\n\n")
        else:
            winner = "EMPATE"
            f.write(f"🤝 EMPATE: Ambos algoritmos encontraron caminos de igual longitud\n\n")

        # Análisis teórico
        f.write("5. ANÁLISIS: ¿Cuál algoritmo encontró el camino más corto?\n")
        f.write("-" * 70 + "\n\n")

        f.write("RESPUESTA:\n")
        f.write(f"El algoritmo BFS (Breadth-First Search) encontró el camino más corto\n")
        f.write(f"con {len(bfs_path) - 1} saltos.\n\n")

        f.write("¿POR QUÉ BFS GARANTIZA EL CAMINO MÁS CORTO?\n\n")

        f.write("1. ESTRATEGIA DE EXPLORACIÓN:\n")
        f.write("   BFS explora el grafo por NIVELES (anchura), visitando primero todos\n")
        f.write("   los nodos a distancia 1, luego todos a distancia 2, y así sucesivamente.\n")
        f.write("   Esto garantiza que cuando BFS encuentra el objetivo, ha encontrado el\n")
        f.write("   camino con menor número de saltos.\n\n")

        f.write("   Exploración de BFS (por niveles):\n")
        f.write("   Nivel 0: Start\n")
        f.write("   Nivel 1: A, B\n")
        f.write("   Nivel 2: C, D, E\n")
        f.write("   Nivel 3: Goal, F\n\n")

        f.write("2. ESTRUCTURA DE DATOS:\n")
        f.write("   BFS usa una COLA (FIFO - First In, First Out), lo que asegura que\n")
        f.write("   los nodos se procesen en el orden en que se descubren. Esto mantiene\n")
        f.write("   la exploración sistemática por niveles.\n\n")

        f.write("3. COMPLETITUD Y OPTIMALIDAD:\n")
        f.write("   • Completitud: BFS siempre encuentra una solución si existe\n")
        f.write("   • Optimalidad: BFS SIEMPRE encuentra el camino más corto en grafos\n")
        f.write("                  no ponderados (donde todos los arcos tienen peso 1)\n\n")

        f.write("¿POR QUÉ DFS NO GARANTIZA EL CAMINO MÁS CORTO?\n\n")

        f.write("1. ESTRATEGIA DE EXPLORACIÓN:\n")
        f.write("   DFS explora el grafo en PROFUNDIDAD, yendo tan lejos como sea posible\n")
        f.write("   por una rama antes de retroceder. Esto significa que DFS puede encontrar\n")
        f.write("   primero un camino largo antes de explorar caminos más cortos.\n\n")

        f.write("2. ESTRUCTURA DE DATOS:\n")
        f.write("   DFS usa una PILA (LIFO - Last In, First Out), lo que hace que explore\n")
        f.write("   profundamente una rama antes de considerar alternativas.\n\n")

        f.write("3. COMPLETITUD SIN OPTIMALIDAD:\n")
        f.write("   • Completitud: DFS encuentra una solución si existe (en grafos finitos)\n")
        f.write("   • Optimalidad: DFS NO garantiza el camino más corto\n\n")

        f.write("EJEMPLO EN NUESTRO GRAFO:\n")
        f.write(f"   BFS encontró: {' → '.join(bfs_path)} ({len(bfs_path) - 1} saltos)\n")
        f.write(f"   DFS encontró: {' → '.join(dfs_path)} ({len(dfs_path) - 1} saltos)\n\n")

        if len(bfs_path) < len(dfs_path):
            f.write(f"   Como se puede ver, BFS encontró un camino {len(dfs_path) - len(bfs_path)} ")
            f.write("salto(s) más corto.\n")
            f.write("   Esto confirma que BFS es óptimo para encontrar caminos más cortos.\n\n")
        else:
            f.write("   En este caso particular, ambos encontraron el mismo camino óptimo,\n")
            f.write("   pero esto es coincidencia - solo BFS lo GARANTIZA siempre.\n\n")

        f.write("6. CONCLUSIONES\n")
        f.write("-" * 70 + "\n")
        f.write("• Usa BFS cuando necesites el camino MÁS CORTO (número mínimo de saltos)\n")
        f.write("• Usa DFS cuando solo necesites ENCONTRAR un camino (no el más corto)\n")
        f.write("• BFS consume más memoria pero garantiza optimalidad\n")
        f.write("• DFS consume menos memoria pero puede encontrar caminos subóptimos\n\n")

        f.write("=" * 70 + "\n")
        f.write("FIN DEL ANÁLISIS\n")
        f.write("=" * 70 + "\n")

    print(f"\n   ✅ Análisis guardado: ANALISIS_BFS_DFS.txt")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Ejecuta el ejercicio completo.
    """
    # Crear grafo
    G = create_graph(warehouse_map)

    print(f"Grafo creado:")
    print(f"  • Nodos: {G.number_of_nodes()}")
    print(f"  • Arcos: {G.number_of_edges()}")
    print(f"  • Objetivo: Encontrar camino de 'Start' a 'Goal'\n")

    print("=" * 70)

    # Ejecutar BFS
    bfs_path, bfs_visited, bfs_order = bfs_from_scratch(G, 'Start', 'Goal')

    print("=" * 70)

    # Ejecutar DFS
    dfs_path, dfs_visited, dfs_order = dfs_from_scratch(G, 'Start', 'Goal')

    print("\n" + "=" * 70)
    print("GENERANDO VISUALIZACIONES...")
    print("=" * 70)

    # Visualizaciones individuales
    visualize_search_algorithm(G, bfs_path, bfs_visited, bfs_order,
                               'BFS', '01_bfs_detallado.png')

    visualize_search_algorithm(G, dfs_path, dfs_visited, dfs_order,
                               'DFS', '02_dfs_detallado.png')

    # Grafo original
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue',
            node_size=2500, font_size=12, font_weight='bold',
            edge_color='gray', width=2, alpha=0.7, edgecolors='black', linewidths=2)
    ax.set_title('Grafo Original de la Bodega', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '00_grafo_original.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n   ✅ Grafo original guardado: 00_grafo_original.png")

    # Comparación lado a lado
    visualize_comparison(G, bfs_path, bfs_visited, dfs_path, dfs_visited)

    # Crear reporte de análisis
    print("\n" + "=" * 70)
    print("GENERANDO ANÁLISIS...")
    print("=" * 70)
    create_analysis_report(G, bfs_path, bfs_visited, bfs_order,
                           dfs_path, dfs_visited, dfs_order)

    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS")
    print("=" * 70)
    print(f"\n🔵 BFS:")
    print(f"   Camino: {' → '.join(bfs_path)}")
    print(f"   Saltos: {len(bfs_path) - 1}")
    print(f"   Nodos visitados: {len(bfs_visited)}/{G.number_of_nodes()}")

    print(f"\n🔴 DFS:")
    print(f"   Camino: {' → '.join(dfs_path)}")
    print(f"   Saltos: {len(dfs_path) - 1}")
    print(f"   Nodos visitados: {len(dfs_visited)}/{G.number_of_nodes()}")

    print(f"\n🏆 Camino más corto: ", end="")
    if len(bfs_path) < len(dfs_path):
        print(f"BFS ({len(bfs_path) - 1} saltos)")
    elif len(dfs_path) < len(bfs_path):
        print(f"DFS ({len(dfs_path) - 1} saltos)")
    else:
        print(f"EMPATE ({len(bfs_path) - 1} saltos)")

    print(f"\n📁 Todos los archivos guardados en: {OUTPUT_DIR}")
    print("   • 00_grafo_original.png")
    print("   • 01_bfs_detallado.png")
    print("   • 02_dfs_detallado.png")
    print("   • 04_comparacion_bfs_dfs.png")
    print("   • ANALISIS_BFS_DFS.txt")

    print("\n🎉 ¡Ejercicio 2.2 completado exitosamente!\n")


if __name__ == "__main__":
    main()