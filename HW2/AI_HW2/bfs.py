import csv
from unittest.mock import NonCallableMagicMock
edgeFile = 'edges.csv'


def bfs(start, end):
    # Begin your code (Part 1)
    '''
    This part is to input the edge data, which is all the same in the 5  route finding algorithms.
    Create a graph using the dictionary structure. The key can be one of the node in the map and 
    will point to a numpy array that contain all edges connected and their detail information. 
    '''
    fptr=open(edgeFile,'r')
    fptr.readline()
    graph={}
    neighbors=[]
    while 1:
        temp=fptr.readline()
        if not temp:
            break
        temp=temp.split(',')
        key=int(temp[0])
        if key not in graph.keys():
            neighbors=[]
        neighbors.append([int(temp[1]), float(temp[2])])
        graph[key]=neighbors
    fptr.close()
    '''
    This is the bfs part. Use numpy array as a queue to store nodes, a set structure to 
    record visited nodes and a dictionary to map the node with its parent node. When there's node 
    in the queue, pop it out, and browse all the nodes connected to it at the same time if
    they are not visited nor the end node. Due to the first in first out property of queue,
    we can just push the node in directly to attain our goal. Finally, browse back from the end
    node to start node by the parent dictionary to store the whole path node as a set and also 
    calculate the total distance of the path.
    '''
    n=0
    queue=[]
    queue.append(start)
    seen=set()
    seen.add(start)
    parent={start:None}
    while len(queue):
        node=queue.pop(0)
        n+=1
        nei=graph.get(node,-1)
        if nei==-1:
            continue
        for x in nei:
            if x[0]==end:
                n+=1
                parent[x[0]]=node
                break
            elif not (x[0] in seen):
                queue.append(x[0])
                seen.add(x[0])
                parent[x[0]]=node
        if (end in parent):
            break
    
    path=[]
    v=end
    dist=0
    while 1:
        path.insert(0,v)
        u=parent[v]
        # print(u, v)
        if u==None:
            break
        for x in graph.get(u):
            if x[0]==v:
                dist+=x[1]
                break
        v=u
    num_visited=n
    return path, dist, num_visited 
    # raise NotImplementedError("To be implemented")
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    # path, dist, num_visited = bfs(26059311, 1924125174)
    # path, dist, num_visited = bfs(26059311, 4419128653)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
    
        