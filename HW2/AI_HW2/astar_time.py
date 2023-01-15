import csv
import heapq
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'

'''
Class in astar(time) is modified to contain 3 variables, which are id, 
cost(time spend from start to node) and
heuristic(time spend from node to end by euclidean dist and the max speed limit of the map)
'''
class Node(object):
    def __init__(self, Id:int,t:float,hVal: float):
        self.t = t
        self.Id = Id
        self.hVal = hVal

    def __repr__(self):
        return f'Node id: {self.Id} Node t: {self.t} Node g+h: {self.hVal+self.t}'

    def __lt__(self, other):
        return self.t+self.hVal < other.t+other.hVal

def astar_time(start, end):
    # Begin your code (Part 6)
    fptr=open(edgeFile,'r')
    fptr.readline()
    graph={}
    neighbors=[]
    speed=0
    while 1:
        temp=fptr.readline()
        if not temp:
            break
        temp=temp.split(',')
        key=int(temp[0])
        if key not in graph.keys():
            neighbors=[]
        neighbors.append([int(temp[1]), float(temp[2]),float(temp[3])])
        speed=max(speed,float(temp[3]))
        graph[key]=neighbors
    fptr.close()

    heu={}
    fptr=open(heuristicFile,'r')
    nID=fptr.readline().split(',')
    for i in range(1,4):
        heu[int(nID[i])]={}
   
    while 1:
        temp=fptr.readline()
        if not temp:
            break
        temp=temp.split(',')
        for i in range(1,4):
            heu[int(nID[i])][int(temp[0])]=float(temp[i])
    fptr.close()
    '''
    This is astar(time) part. The comparison factor become 
    cost(time spend from start to node) +
    heuristic(time spend from node to end by euclidean dist and the max speed limit of the map)
    here. For the Heuristic part, I calculate the time this way to ensure that its admissible,
    and we can get the max speed limit while inputting the edges data.
    '''
    n=0
    pq=[]
    heapq.heappush(pq,Node(start,0,heu[end][start]/speed))
    seen=set()
    parent={Node(start,0,heu[end][start]/speed):None}
    while len(pq):
        node=heapq.heappop(pq)
        # print(node)
        if node.Id in seen:
            continue
        n+=1
        if node.Id==end:
            time=node.t
            time*=3.6
            v=node
            break
        seen.add(node.Id)
        nei=graph.get(node.Id,-1)
        if nei==-1 :
            continue
        for x in nei:
            if not(x[0] in seen):
                temp=Node(x[0],(node.t+x[1]/x[2]),heu[end][x[0]]/speed) 
                heapq.heappush(pq,temp)
                parent[temp]=node

    path=[]
    while v!=None:
        path.insert(0,v.Id)
        v=parent.get(v)
    num_visited=n
    return path, time, num_visited
    # raise NotImplementedError("To be implemented")
    # End your code (Part 6)


if __name__ == '__main__':
    path, time, num_visited = astar_time(2270143902, 1079387396)
    # path, time, num_visited = astar_time(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
