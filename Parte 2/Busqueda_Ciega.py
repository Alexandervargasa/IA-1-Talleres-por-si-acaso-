"""
Parte 2: Formulación de Problemas y Búsqueda Ciega
Sistema de Navegación para Robot en Bodega usando Grafos
VERSIÓN COMPATIBLE CON WINDOWS/LINUX/MAC
"""

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Dict, Tuple, Optional
import os

# ============================================================================
# CONFIGURACIÓN DE RUTAS MULTIPLATAFORMA
# ============================================================================

# Crear carpeta de salida en el directorio actual si no existe
OUTPUT_DIR = os.path.join(os.getcwd(), 'resultados_busqueda')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"📁 Guardando resultados en: {OUTPUT_DIR}")

# ============================================================================
# EJERCICIO 2.1: Definición del Espacio de Estados
# ============================================================================

# Definición del mapa de la bodega como diccionario de adyacencias
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


def create_warehouse_graph(adjacency_dict: Dict[str, List[str]]) -> nx.Graph:
    """
    Crea un grafo de NetworkX a partir del diccionario de adyacencias.

    Args:
        adjacency_dict: Diccionario con nodos y sus conexiones

    Returns:
        Grafo de NetworkX
    """
    G = nx.Graph()

    # Agregar nodos
    for node in adjacency_dict.keys():
        G.add_node(node)

    # Agregar arcos (conexiones transitables)
    for node, neighbors in adjacency_dict.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    return G


def visualize_graph(G: nx.Graph, path: Optional[List[str]] = None,
                    title: str = "Mapa de la Bodega", filename: str = "warehouse_graph.png"):
    """
    Visualiza el grafo de la bodega.

    Args:
        G: Grafo de NetworkX
        path: Camino a resaltar (opcional)
        title: Título del gráfico
        filename: Nombre del archivo a guardar
    """
    plt.figure(figsize=(12, 8))

    # Layout para mejor visualización
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

    # Dibujar todos los nodos
    node_colors = []
    for node in G.nodes():
        if node == 'Start':
            node_colors.append('lightgreen')
        elif node == 'Goal':
            node_colors.append('lightcoral')
        elif path and node in path:
            node_colors.append('yellow')
        else:
            node_colors.append('lightblue')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=2000, alpha=0.9)

    # Dibujar arcos
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')

    # Si hay un camino, resaltarlo
    if path:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                               width=4, alpha=0.8, edge_color='red')

    # Dibujar etiquetas
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    # Guardar en el directorio de salida
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Gráfico guardado: {output_path}")
    plt.close()


# ============================================================================
# BÚSQUEDA EN ANCHURA (BFS - Breadth-First Search)
# ============================================================================

def bfs_search(graph: nx.Graph, start: str, goal: str) -> Tuple[Optional[List[str]], Dict]:
    """
    Implementación de BFS para encontrar el camino más corto.

    Args:
        graph: Grafo de NetworkX
        start: Nodo inicial
        goal: Nodo objetivo

    Returns:
        Tupla con (camino encontrado, estadísticas de búsqueda)
    """
    # Cola FIFO para BFS
    queue = deque([(start, [start])])
    visited = set([start])
    nodes_explored = 0

    while queue:
        current_node, path = queue.popleft()
        nodes_explored += 1

        # ¿Llegamos al objetivo?
        if current_node == goal:
            stats = {
                'nodes_explored': nodes_explored,
                'path_length': len(path),
                'algorithm': 'BFS'
            }
            return path, stats

        # Explorar vecinos
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    # No se encontró camino
    return None, {'nodes_explored': nodes_explored, 'algorithm': 'BFS'}


# ============================================================================
# BÚSQUEDA EN PROFUNDIDAD (DFS - Depth-First Search)
# ============================================================================

def dfs_search(graph: nx.Graph, start: str, goal: str) -> Tuple[Optional[List[str]], Dict]:
    """
    Implementación de DFS para encontrar un camino.

    Args:
        graph: Grafo de NetworkX
        start: Nodo inicial
        goal: Nodo objetivo

    Returns:
        Tupla con (camino encontrado, estadísticas de búsqueda)
    """
    # Pila LIFO para DFS
    stack = [(start, [start])]
    visited = set()
    nodes_explored = 0

    while stack:
        current_node, path = stack.pop()

        if current_node in visited:
            continue

        visited.add(current_node)
        nodes_explored += 1

        # ¿Llegamos al objetivo?
        if current_node == goal:
            stats = {
                'nodes_explored': nodes_explored,
                'path_length': len(path),
                'algorithm': 'DFS'
            }
            return path, stats

        # Explorar vecinos (en orden inverso para mantener consistencia)
        neighbors = list(graph.neighbors(current_node))
        for neighbor in reversed(neighbors):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    # No se encontró camino
    return None, {'nodes_explored': nodes_explored, 'algorithm': 'DFS'}


# ============================================================================
# FUNCIÓN PRINCIPAL DE DEMOSTRACIÓN
# ============================================================================

def main():
    """
    Función principal que ejecuta todos los algoritmos y genera reportes.
    """
    print("=" * 70)
    print("SISTEMA DE NAVEGACIÓN PARA ROBOT EN BODEGA")
    print("Búsqueda Ciega: BFS vs DFS")
    print("=" * 70)

    # Crear el grafo
    print("\n1. Creando grafo de la bodega...")
    G = create_warehouse_graph(warehouse_map)

    print(f"   - Nodos (zonas): {G.number_of_nodes()}")
    print(f"   - Arcos (conexiones): {G.number_of_edges()}")
    print(f"   - Zonas: {list(G.nodes())}")

    # Visualizar grafo original
    print("\n2. Generando visualización del grafo...")
    visualize_graph(G, title="Mapa de la Bodega - Todas las Conexiones",
                    filename="01_grafo_completo.png")

    # Ejecutar BFS
    print("\n3. Ejecutando BFS (Breadth-First Search)...")
    bfs_path, bfs_stats = bfs_search(G, 'Start', 'Goal')

    if bfs_path:
        print(f"   ✓ Camino encontrado: {' → '.join(bfs_path)}")
        print(f"   - Nodos explorados: {bfs_stats['nodes_explored']}")
        print(f"   - Longitud del camino: {bfs_stats['path_length']}")

        # Visualizar camino BFS
        visualize_graph(G, path=bfs_path,
                        title=f"BFS: Camino más corto ({bfs_stats['path_length']} pasos)",
                        filename="02_bfs_resultado.png")
    else:
        print("   ✗ No se encontró camino")

    # Ejecutar DFS
    print("\n4. Ejecutando DFS (Depth-First Search)...")
    dfs_path, dfs_stats = dfs_search(G, 'Start', 'Goal')

    if dfs_path:
        print(f"   ✓ Camino encontrado: {' → '.join(dfs_path)}")
        print(f"   - Nodos explorados: {dfs_stats['nodes_explored']}")
        print(f"   - Longitud del camino: {dfs_stats['path_length']}")

        # Visualizar camino DFS
        visualize_graph(G, path=dfs_path,
                        title=f"DFS: Camino encontrado ({dfs_stats['path_length']} pasos)",
                        filename="03_dfs_resultado.png")
    else:
        print("   ✗ No se encontró camino")

    # Comparación
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE ALGORITMOS")
    print("=" * 70)

    if bfs_path and dfs_path:
        print(f"\n🔵 BFS (Breadth-First Search):")
        print(f"  - Camino: {' → '.join(bfs_path)}")
        print(f"  - Longitud: {bfs_stats['path_length']} pasos")
        print(f"  - Nodos explorados: {bfs_stats['nodes_explored']}")
        print(f"  - Garantía: Camino MÁS CORTO ✓")

        print(f"\n🔴 DFS (Depth-First Search):")
        print(f"  - Camino: {' → '.join(dfs_path)}")
        print(f"  - Longitud: {dfs_stats['path_length']} pasos")
        print(f"  - Nodos explorados: {dfs_stats['nodes_explored']}")
        print(f"  - Garantía: Encuentra un camino (no necesariamente el más corto)")

        if bfs_stats['path_length'] < dfs_stats['path_length']:
            print(
                f"\n✅ BFS encontró un camino más corto ({bfs_stats['path_length']} vs {dfs_stats['path_length']} pasos)")
        elif bfs_stats['path_length'] == dfs_stats['path_length']:
            print(f"\n✅ Ambos encontraron caminos de igual longitud ({bfs_stats['path_length']} pasos)")
        else:
            print(f"\n⚠️ DFS encontró un camino más corto (caso raro)")

    # Crear reporte de texto
    print("\n5. Generando reporte de texto...")
    create_text_report(G, bfs_path, bfs_stats, dfs_path, dfs_stats)

    print("\n" + "=" * 70)
    print("ANÁLISIS TEÓRICO")
    print("=" * 70)
    print("""
🔵 BFS (Breadth-First Search):
  • Estrategia: Explora por niveles (anchura)
  • Estructura: Cola FIFO (First In, First Out)
  • Completitud: Sí (siempre encuentra solución si existe)
  • Optimalidad: Sí (encuentra el camino más corto)
  • Complejidad temporal: O(V + E)
  • Complejidad espacial: O(V)

🔴 DFS (Depth-First Search):
  • Estrategia: Explora en profundidad primero
  • Estructura: Pila LIFO (Last In, First Out)
  • Completitud: Sí (en grafos finitos)
  • Optimalidad: No (puede encontrar caminos más largos)
  • Complejidad temporal: O(V + E)
  • Complejidad espacial: O(V)

Donde V = número de vértices (nodos) y E = número de aristas (arcos)
    """)

    print(f"\n📁 Archivos generados en: {OUTPUT_DIR}")
    print("   - 01_grafo_completo.png")
    print("   - 02_bfs_resultado.png")
    print("   - 03_dfs_resultado.png")
    print("   - reporte_busqueda.txt")
    print("\n🎉 ¡Búsqueda completada exitosamente! 🤖\n")


def create_text_report(G, bfs_path, bfs_stats, dfs_path, dfs_stats):
    """
    Crea un reporte de texto con los resultados.
    """
    report_path = os.path.join(OUTPUT_DIR, "reporte_busqueda.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("REPORTE DE BÚSQUEDA - NAVEGACIÓN EN BODEGA\n")
        f.write("=" * 70 + "\n\n")

        f.write("INFORMACIÓN DEL GRAFO\n")
        f.write("-" * 70 + "\n")
        f.write(f"Nodos (zonas): {G.number_of_nodes()}\n")
        f.write(f"Arcos (conexiones): {G.number_of_edges()}\n")
        f.write(f"Zonas: {list(G.nodes())}\n\n")

        f.write("DICCIONARIO DE ADYACENCIAS\n")
        f.write("-" * 70 + "\n")
        for node in sorted(G.nodes()):
            neighbors = list(G.neighbors(node))
            f.write(f"'{node}': {neighbors}\n")
        f.write("\n")

        f.write("RESULTADOS DE BFS (Breadth-First Search)\n")
        f.write("-" * 70 + "\n")
        if bfs_path:
            f.write(f"Camino encontrado: {' → '.join(bfs_path)}\n")
            f.write(f"Longitud del camino: {bfs_stats['path_length']} pasos\n")
            f.write(f"Nodos explorados: {bfs_stats['nodes_explored']}\n")
            f.write("Garantía: Camino MÁS CORTO ✓\n\n")
        else:
            f.write("No se encontró camino\n\n")

        f.write("RESULTADOS DE DFS (Depth-First Search)\n")
        f.write("-" * 70 + "\n")
        if dfs_path:
            f.write(f"Camino encontrado: {' → '.join(dfs_path)}\n")
            f.write(f"Longitud del camino: {dfs_stats['path_length']} pasos\n")
            f.write(f"Nodos explorados: {dfs_stats['nodes_explored']}\n")
            f.write("Garantía: Encuentra un camino (no necesariamente el más corto)\n\n")
        else:
            f.write("No se encontró camino\n\n")

        f.write("COMPARACIÓN\n")
        f.write("-" * 70 + "\n")
        if bfs_path and dfs_path:
            f.write(f"BFS - Longitud: {bfs_stats['path_length']} | Nodos explorados: {bfs_stats['nodes_explored']}\n")
            f.write(f"DFS - Longitud: {dfs_stats['path_length']} | Nodos explorados: {dfs_stats['nodes_explored']}\n\n")

            if bfs_stats['path_length'] < dfs_stats['path_length']:
                f.write(f"Ganador: BFS (camino más corto)\n")
            elif bfs_stats['path_length'] == dfs_stats['path_length']:
                f.write(f"Empate: Ambos encontraron caminos de igual longitud\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("=" * 70 + "\n")

    print(f"   ✅ Reporte guardado: {report_path}")


# ============================================================================
# EJERCICIOS ADICIONALES (BONUS)
# ============================================================================

def generate_random_warehouse(num_zones: int = 15, edge_probability: float = 0.3) -> Dict[str, List[str]]:
    """
    Genera un mapa de bodega aleatorio más complejo.

    Args:
        num_zones: Número de zonas en la bodega
        edge_probability: Probabilidad de conexión entre nodos

    Returns:
        Diccionario de adyacencias
    """
    import random

    # Crear grafo aleatorio
    G = nx.erdos_renyi_graph(num_zones, edge_probability, seed=42)

    # Convertir a diccionario de adyacencias
    adjacency_dict = {}
    mapping = {i: f'Z{i}' for i in range(num_zones)}
    mapping[0] = 'Start'
    mapping[num_zones - 1] = 'Goal'

    G = nx.relabel_nodes(G, mapping)

    for node in G.nodes():
        adjacency_dict[node] = list(G.neighbors(node))

    return adjacency_dict


if __name__ == "__main__":
    main()

    # BONUS: Probar con grafo más complejo
    print("\n" + "=" * 70)
    print("BONUS: Prueba con Bodega Aleatoria Más Compleja")
    print("=" * 70)

    random_warehouse = generate_random_warehouse(num_zones=12, edge_probability=0.25)
    G_random = create_warehouse_graph(random_warehouse)

    print(f"\nGrafo aleatorio generado:")
    print(f"  - Nodos: {G_random.number_of_nodes()}")
    print(f"  - Arcos: {G_random.number_of_edges()}")

    if nx.has_path(G_random, 'Start', 'Goal'):
        print("\n✅ Existe camino entre Start y Goal")
        bfs_path_r, bfs_stats_r = bfs_search(G_random, 'Start', 'Goal')
        dfs_path_r, dfs_stats_r = dfs_search(G_random, 'Start', 'Goal')

        print(f"  - BFS encontró camino de longitud: {bfs_stats_r['path_length']}")
        print(f"  - DFS encontró camino de longitud: {dfs_stats_r['path_length']}")

        # Visualizar grafo aleatorio
        visualize_graph(G_random, path=bfs_path_r,
                        title="Grafo Aleatorio - Camino BFS",
                        filename="04_grafo_aleatorio.png")
    else:
        print("\n⚠️ No hay camino entre Start y Goal en este grafo aleatorio")
        visualize_graph(G_random, title="Grafo Aleatorio - Sin Camino",
                        filename="04_grafo_aleatorio.png")

    print(f"\n🎉 ¡Proceso completo finalizado! Revisa la carpeta: {OUTPUT_DIR}\n")