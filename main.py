import PySimpleGUI as sg
import heapq
from typing import List, Tuple, Optional, Set
import sys
from collections import defaultdict

# Algorithm descriptions and help text
ALGORITHM_HELP = {
    'Nearest Neighbor': {
        'title': 'Nearest Neighbor Algorithm Help',
        'desc': [
            '• Values represent distances between nodes/cities',
            '• Diagonal elements must be 0 (no self-loops)',
            '• Empty cells will be treated as no connection (0)',
            '• Matrix should be symmetric (distance A→B equals B→A)',
            '• All distances must be positive'
        ]
    },
    'Best Edge': {
        'title': 'Best Edge Algorithm Help',
        'desc': [
            '• Values represent distances between nodes/cities',
            '• Diagonal elements must be 0 (no self-loops)',
            '• Empty cells will be treated as no connection (0)',
            '• Matrix should be symmetric (distance A→B equals B→A)',
            '• All distances must be positive'
        ]
    },
    'Dijkstra\'s': {
        'title': 'Dijkstra\'s Algorithm Help',
        'desc': [
            '• Values represent path weights/distances',
            '• Diagonal elements can be any non-negative value',
            '• Empty cells will be treated as no connection (0)',
            '• Matrix can be asymmetric (one-way paths allowed)',
            '• All weights must be positive'
        ]
    },
    'Kruskal\'s': {
        'title': 'Kruskal\'s Algorithm Help',
        'desc': [
            '• Values represent connection weights',
            '• Diagonal elements can be any non-negative value',
            '• Empty cells will be treated as no connection (0)',
            '• Matrix should be symmetric (undirected edges)',
            '• All weights must be positive'
        ]
    },
    'Critical Path': {
        'title': 'Critical Path Algorithm Help',
        'desc': [
            '• Values represent task dependencies and durations',
            '• Diagonal elements must be 0 (no self-dependencies)',
            '• Empty cells will be treated as no dependency (0)',
            '• Matrix must be asymmetric (directed acyclic graph)',
            '• All durations must be positive'
        ]
    }
}

def show_help(algorithm_name: str):
    """Display help window for specific algorithm"""
    help_info = ALGORITHM_HELP[algorithm_name]
    layout = [
        [sg.Text(help_info['title'], font=("Helvetica", 12, 'bold'), text_color='#1a237e')],
        [sg.Text('_' * 50)],
        *[[sg.Text(desc)] for desc in help_info['desc']],
        [sg.Button('OK', button_color=('#ffffff', '#1a237e'))]
    ]
    
    window = sg.Window(
        'Algorithm Help',
        layout,
        font=("Helvetica", 10),
        element_justification='left',
        modal=True
    )
    
    window.read()
    window.close()

def create_matrix_input_window(size: int, title: str, algorithm_name: str) -> sg.Window:
    """Creates an enhanced window with algorithm-specific instructions"""
    sg.theme('LightGrey1')
    
    help_info = ALGORITHM_HELP[algorithm_name]
    
    header_layout = [
        [sg.Text("Matrix Input Guide:", font=("Helvetica", 12, 'bold'), text_color='#1a237e')],
        *[[sg.Text(desc, font=("Helvetica", 10))] for desc in help_info['desc'][:3]],
        [sg.Button('Show Full Help', key='-HELP-', 
                  button_color=('#ffffff', '#1a237e'),
                  font=("Helvetica", 10))]
    ]

    matrix_layout = [
        [sg.Text("To →", size=(4,1))] + 
        [sg.Text(f"Node {j}", size=(5,1), justification='center', background_color='#dcdcdc') 
         for j in range(size)]
    ]

    needs_zero_diagonal = algorithm_name in ['Nearest Neighbor', 'Best Edge', 'Critical Path']

    for i in range(size):
        row = [sg.Text(f"From {i}", size=(8,1), justification='right', background_color='#dcdcdc')]
        for j in range(size):
            if i == j and needs_zero_diagonal:
                row.append(sg.Input('0', size=(5, 1), 
                          key=f'cell_{i}_{j}',
                          justification='center',
                          background_color='#e3f2fd',
                          tooltip='Must be 0 for this algorithm',
                          disabled=True))
            else:
                row.append(sg.Input(size=(5, 1), 
                          key=f'cell_{i}_{j}',
                          justification='center',
                          tooltip=f'Enter weight/distance from Node {i} to Node {j}\nLeave empty for no connection'))
        matrix_layout.append(row)

    button_style = {
        'size': (10, 1),
        'font': ('Helvetica', 10),
        'button_color': ('#ffffff', '#1a237e')
    }

    layout = [
        [sg.Frame("Instructions", header_layout, font=("Helvetica", 12, 'bold'))],
        [sg.Column(matrix_layout, element_justification='center', pad=(10,10))],
        [sg.Button("Submit", **button_style), 
         sg.Button("Clear", **button_style), 
         sg.Button("Back", **button_style)]
    ]

    window = sg.Window(
        title, 
        layout,
        finalize=True,
        font=("Helvetica", 10),
        element_padding=(5, 5),
        margins=(10, 10)
    )

    def clear_inputs():
        for i in range(size):
            for j in range(size):
                if not (i == j and needs_zero_diagonal):
                    window[f'cell_{i}_{j}'].update('')

    window.bind('<Clear>', clear_inputs)
    
    return window

def get_matrix_from_window(window: sg.Window, size: int, algorithm: str) -> Optional[List[List[int]]]:
    """Gets and validates matrix with algorithm-specific rules"""
    try:
        matrix = []
        needs_zero_diagonal = algorithm in ['Nearest Neighbor', 'Best Edge', 'Critical Path']
        needs_symmetric = algorithm in ['Nearest Neighbor', 'Best Edge', 'Kruskal\'s']
        
        for i in range(size):
            row = []
            for j in range(size):
                value = window[f'cell_{i}_{j}'].get().strip()
                
                # Handle empty cells
                if value == '':
                    value = '0'
                    
                value = int(value)
                
                # Validate based on algorithm requirements
                if value < 0:
                    raise ValueError("Negative values are not allowed")
                    
                if i == j and needs_zero_diagonal and value != 0:
                    raise ValueError(f"Diagonal elements must be 0 for {algorithm}")
                    
                row.append(value)
            matrix.append(row)
            
        # Check symmetry if required
        if needs_symmetric:
            for i in range(size):
                for j in range(i+1, size):
                    if matrix[i][j] != matrix[j][i]:
                        raise ValueError(f"{algorithm} requires symmetric values (same distance both ways)")
                        
        return matrix
    except ValueError as e:
        sg.popup_error(f"Invalid input: {str(e)}")
        return None
    
def nearest_neighbor(adjacency_matrix: List[List[int]], start_point: int) -> str:
    """Implements Nearest Neighbor Algorithm"""
    n = len(adjacency_matrix)
    if start_point < 0 or start_point >= n:
        return "Error: Start point is not valid."

    visited = [False] * n
    path = []
    total_weight = 0

    current_node = start_point
    visited[current_node] = True
    path.append(current_node)

    for _ in range(n - 1):
        min_distance = float('inf')
        next_node = None

        for neighbor in range(n):
            if not visited[neighbor] and 0 < adjacency_matrix[current_node][neighbor] < min_distance:
                min_distance = adjacency_matrix[current_node][neighbor]
                next_node = neighbor

        if next_node is not None:
            visited[next_node] = True
            path.append(next_node)
            total_weight += min_distance
            current_node = next_node
        else:
            return "Error: The graph is disconnected. Not all nodes can be visited."

    if adjacency_matrix[current_node][start_point] > 0:
        total_weight += adjacency_matrix[current_node][start_point]
        path.append(start_point)
    else:
        return "Error: Cannot return to the start point."

    return f"Total Path Weight: {total_weight}\nPath: {' → '.join(map(str, path))}"

def best_edge(graph: List[List[int]], start: int) -> Tuple[int, List[int]]:
    """Implements Best Edge Algorithm for TSP"""
    n = len(graph)
    visited = [False] * n
    visited[start] = True
    path = [start]
    total_weight = 0
    current = start

    while len(path) < n:
        best_weight = float('inf')
        best_next = -1
        
        for next_node in range(n):
            if not visited[next_node] and graph[current][next_node] > 0:
                if graph[current][next_node] < best_weight:
                    best_weight = graph[current][next_node]
                    best_next = next_node
        
        if best_next == -1:
            return -1, []  # No valid path found
            
        visited[best_next] = True
        path.append(best_next)
        total_weight += best_weight
        current = best_next

    # Return to start
    if graph[current][start] > 0:
        total_weight += graph[current][start]
        path.append(start)
        return total_weight, path
    return -1, []

def dijkstra(graph: List[List[int]], start: int, end: int) -> Tuple[int, List[int]]:
    """Implements Dijkstra's shortest path algorithm"""
    n = len(graph)
    distances = [float('inf')] * n
    distances[start] = 0
    previous = [None] * n
    pq = [(0, start)]
    visited = set()

    while pq:
        current_distance, current = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == end:
            break
            
        for neighbor in range(n):
            if graph[current][neighbor] > 0:
                distance = current_distance + graph[current][neighbor]
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
    
    if distances[end] == float('inf'):
        return -1, []
        
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    return distances[end], path

def kruskal(graph: List[List[int]]) -> List[List[int]]:
    """Implements Kruskal's minimum spanning tree algorithm"""
    n = len(graph)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if graph[i][j] > 0:
                edges.append((graph[i][j], i, j))
    
    edges.sort()
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        parent[find(x)] = find(y)
    
    mst = [[0] * n for _ in range(n)]
    for weight, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst[u][v] = mst[v][u] = weight
    
    return mst

def critical_path(graph: List[List[int]]) -> List[int]:
    """Implements Critical Path Algorithm"""
    n = len(graph)
    
    # Calculate earliest start times
    earliest = [0] * n
    for i in range(n):
        for j in range(n):
            if graph[j][i] > 0:
                earliest[i] = max(earliest[i], earliest[j] + graph[j][i])
    
    # Calculate latest start times
    project_duration = max(earliest)
    latest = [project_duration] * n
    
    for i in range(n-1, -1, -1):
        for j in range(n):
            if graph[i][j] > 0:
                latest[i] = min(latest[i], latest[j] - graph[i][j])
    
    # Find critical path
    critical_nodes = []
    for i in range(n):
        if earliest[i] == latest[i]:
            critical_nodes.append(i)
    
    return critical_nodes


class GraphGUI:
    def __init__(self):
        sg.theme('LightGrey1')
        self.window = None
        self.create_main_window()

    def create_main_window(self):
        title_font = ("Helvetica", 24, 'bold')
        button_font = ("Helvetica", 12)
        
        # Header section with watermark
        header = [
            [sg.Text('Graph Algorithm Visualizer', 
                    font=title_font, 
                    text_color='#1a237e',
                    justification='center',
                    expand_x=True,
                    pad=(0, 20))],
            [sg.Text('Created by Shreyank Kulkarni © 2024', 
                    font=("Helvetica", 8),
                    text_color='#666666',
                    justification='right',
                    pad=(0, 0),
                    expand_x=True)]
        ]

        # Algorithm buttons with descriptions
        algorithms = [
            ('Nearest Neighbor', 'Find shortest route visiting all nodes once'),
            ('Best Edge', 'Construct path using locally optimal choices'),
            ('Dijkstra\'s', 'Find shortest path between two nodes'),
            ('Kruskal\'s', 'Find minimum spanning tree'),
            ('Critical Path', 'Find critical path in directed graph')
        ]

        button_layout = []
        for name, desc in algorithms:
            button_layout.append([
                sg.Button(name, 
                         size=(15, 2), 
                         font=button_font,
                         button_color=('#ffffff', '#1a237e'),
                         mouseover_colors=('#ffffff', '#303f9f')),
                sg.Text(desc, font=("Helvetica", 10), pad=(10, 0)),
                sg.Button('?', key=f'-HELP-{name}', 
                         font=button_font,
                         button_color=('#ffffff', '#303f9f'),
                         size=(3, 1))
            ])

        # Footer with exit button and easter egg
        footer = [
            [sg.Button('Exit', 
                      size=(15, 2), 
                      font=button_font,
                      button_color=('#ffffff', '#e91e63'),
                      mouseover_colors=('#ffffff', '#c2185b')),
             sg.Push(),  # Pushes easter egg button to the right
             sg.Button('✨', 
                      key='-EASTER-',
                      size=(3, 1),
                      font=button_font,
                      button_color=('#ffffff', '#9c27b0'),
                      tooltip='What could this be?')]
        ]

        layout = [
            *header,
            [sg.Column(button_layout, element_justification='center', pad=(0, 20))],
            *footer
        ]
        
        self.window = sg.Window(
            'Graph Algorithms',
            layout,
            finalize=True,
            element_justification='center',
            size=(700, 500),
            margins=(20, 20)
        )

    def create_size_input_window(self, algorithm_name: str) -> Tuple[int, bool]:
        """Creates a window to input matrix size with enhanced visuals"""
        layout = [
            [sg.Text(f"Enter number of nodes for {algorithm_name}:", 
                    font=("Helvetica", 12, 'bold'))],
            [sg.Text("This will create an NxN matrix where N is your input", 
                    font=("Helvetica", 10))],
            [sg.Input(size=(5, 1), key='size', justification='center')],
            [sg.Button('Continue', button_color=('#ffffff', '#1a237e')), 
             sg.Button('Back', button_color=('#ffffff', '#e91e63'))]
        ]
        
        window = sg.Window(
            f'{algorithm_name} - Matrix Size', 
            layout,
            element_justification='center',
            font=("Helvetica", 10),
            modal=True
        )
        
        while True:
            event, values = window.read()
            if event in (sg.WINDOW_CLOSED, 'Back'):
                window.close()
                return 0, False
            
            if event == 'Continue':
                try:
                    size = int(values['size'])
                    if size <= 1:
                        raise ValueError
                    window.close()
                    return size, True
                except ValueError:
                    sg.popup_error('Please enter a valid number greater than 1')
                    
        window.close()
        return 0, False

    def display_result(self, title: str, result_text: str):
        """Displays algorithm results in an enhanced window"""
        layout = [
            [sg.Text("Results", font=("Helvetica", 14, 'bold'), text_color='#1a237e')],
            [sg.Text('_' * 50)],
            [sg.Text(result_text, font=('Courier', 12))],
            [sg.Button('OK', button_color=('#ffffff', '#1a237e'))]
        ]
        window = sg.Window(
            title, 
            layout,
            element_justification='center',
            font=("Helvetica", 10),
            modal=True
        )
        window.read()
        window.close()

    def get_start_point(self, size: int) -> Optional[int]:
        """Enhanced window for starting point input"""
        layout = [
            [sg.Text("Select Starting Node", font=("Helvetica", 12, 'bold'))],
            [sg.Text(f"Enter a node number (0-{size-1}):", font=("Helvetica", 10))],
            [sg.Input(size=(5, 1), key='start', justification='center')],
            [sg.Button('Submit', button_color=('#ffffff', '#1a237e')), 
             sg.Button('Cancel', button_color=('#ffffff', '#e91e63'))]
        ]
        
        window = sg.Window(
            'Start Node Selection', 
            layout,
            element_justification='center',
            font=("Helvetica", 10),
            modal=True
        )
        
        while True:
            event, values = window.read()
            if event in (sg.WINDOW_CLOSED, 'Cancel'):
                window.close()
                return None
                
            if event == 'Submit':
                try:
                    start = int(values['start'])
                    if 0 <= start < size:
                        window.close()
                        return start
                    raise ValueError
                except ValueError:
                    sg.popup_error(f'Please enter a valid node number between 0 and {size-1}')
                    
        window.close()
        return None

    def get_start_end_points(self, size: int) -> Tuple[Optional[int], Optional[int]]:
        """Enhanced window for start and end point input"""
        layout = [
            [sg.Text("Select Start and End Nodes", font=("Helvetica", 12, 'bold'))],
            [sg.Text(f"Enter node numbers (0-{size-1}):", font=("Helvetica", 10))],
            [sg.Text("Start Node:", size=(10, 1)), 
             sg.Input(size=(5, 1), key='start', justification='center')],
            [sg.Text("End Node:", size=(10, 1)), 
             sg.Input(size=(5, 1), key='end', justification='center')],
            [sg.Button('Submit', button_color=('#ffffff', '#1a237e')), 
             sg.Button('Cancel', button_color=('#ffffff', '#e91e63'))]
        ]
        
        window = sg.Window(
            'Node Selection', 
            layout,
            element_justification='center',
            font=("Helvetica", 10),
            modal=True
        )
        
        while True:
            event, values = window.read()
            if event in (sg.WINDOW_CLOSED, 'Cancel'):
                window.close()
                return None, None
                
            if event == 'Submit':
                try:
                    start = int(values['start'])
                    end = int(values['end'])
                    if 0 <= start < size and 0 <= end < size and start != end:
                        window.close()
                        return start, end
                    raise ValueError
                except ValueError:
                    sg.popup_error(f'Please enter valid node numbers between 0 and {size-1}')
                    
        window.close()
        return None, None

    def run(self):
        while True:
            event, values = self.window.read()

            if event in (sg.WINDOW_CLOSED, 'Exit'):
                break

            if event == '-EASTER-':
                sg.popup('👋 Hello there!\n\nVisit shreyankk.com\nto see more cool projects!',
                        title='Easter Egg Found!',
                        font=("Helvetica", 12),
                        text_color='#9c27b0',
                        button_color=('#ffffff', '#9c27b0'))
                continue

            if event.startswith('-HELP-'):
                algorithm_name = event.replace('-HELP-', '')
                show_help(algorithm_name)
                continue

            if event in ('Nearest Neighbor', 'Best Edge', 'Dijkstra\'s', 'Kruskal\'s', 'Critical Path'):
                size, proceed = self.create_size_input_window(event)
                
                if not proceed:
                    continue

                matrix_window = create_matrix_input_window(size, f'{event} - Matrix Input', event)
                
                while True:
                    matrix_event, matrix_values = matrix_window.read()
                    
                    if matrix_event == '-HELP-':
                        show_help(event)
                        continue
                    
                    if matrix_event in (sg.WINDOW_CLOSED, 'Back'):
                        matrix_window.close()
                        break

                    if matrix_event == 'Clear':
                        for i in range(size):
                            for j in range(size):
                                if not (i == j and event in ['Nearest Neighbor', 'Best Edge', 'Critical Path']):
                                    matrix_window[f'cell_{i}_{j}'].update('')
                        continue

                    if matrix_event == 'Submit':
                        matrix = get_matrix_from_window(matrix_window, size, event)
                        
                        if matrix is not None:
                            matrix_window.close()
                            
                            if event == 'Nearest Neighbor':
                                start_point = self.get_start_point(size)
                                if start_point is not None:
                                    result = nearest_neighbor(matrix, start_point)
                                    self.display_result('Nearest Neighbor Result', result)

                            elif event == 'Best Edge':
                                start_point = self.get_start_point(size)
                                if start_point is not None:
                                    weight, path = best_edge(matrix, start_point)
                                    if weight != -1:
                                        result = f"Total Weight: {weight}\nPath: {' → '.join(map(str, path))}"
                                    else:
                                        result = "No valid path found"
                                    self.display_result('Best Edge Result', result)

                            elif event == 'Dijkstra\'s':
                                start, end = self.get_start_end_points(size)
                                if start is not None and end is not None:
                                    distance, path = dijkstra(matrix, start, end)
                                    if distance != -1:
                                        result = f"Shortest Distance: {distance}\nPath: {' → '.join(map(str, path))}"
                                    else:
                                        result = "No path found between the selected nodes"
                                    self.display_result('Dijkstra Result', result)

                            elif event == 'Kruskal\'s':
                                mst = kruskal(matrix)
                                result = "Minimum Spanning Tree Edges:\n"
                                for i in range(size):
                                    for j in range(i + 1, size):
                                        if mst[i][j] > 0:
                                            result += f"Node {i} → Node {j}: Weight {mst[i][j]}\n"
                                self.display_result('Kruskal Result', result)

                            elif event == 'Critical Path':
                                critical_nodes = critical_path(matrix)
                                if critical_nodes:
                                    result = f"Critical Path: {' → '.join(map(str, critical_nodes))}"
                                else:
                                    result = "No critical path found"
                                self.display_result('Critical Path Result', result)
                            break

        self.window.close()

# Main execution
if __name__ == '__main__':
    app = GraphGUI()
    app.run()
