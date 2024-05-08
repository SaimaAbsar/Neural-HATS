
# Python program to print all paths from a source to destination. 
# for static data
   
from collections import defaultdict 
import numpy as np
   
# This class represents a directed graph  
# using adjacency list representation 

class AdjNode:
    def __init__(self, V, Dir): 
        # No. of vertices 
        self.v = V  
        self.dir = Dir

OUT = True
IN = False



class PathNode:
    def __init__(self, node, arrow):
        self.node = node
        self.arrow = arrow

    def __rep__(self):
        mstr = str(self.node)+' '+self.arrow
        return mstr

    def __str__(self):
        mstr = str(self.node)+' '+self.arrow
        return mstr


class Graph: 
    def __init__(self, V): 
        # No. of vertices 
        self.V = V  
          
        # default dictionary to store graph 
        self.graph = defaultdict(list)  
        
        self.adj = [ [] for i in range(self.V) ]
        self.ancestor = [ [] for i in range(self.V) ]
        # for latter use
        self.path_index = -1
        self.paths = {}  # entry = (src,dest): [path]
        self.nodeCnt = 0
   
    # prints the recorded paths in self.paths{}
    def printRecordedPaths(self):
        #print(self.paths)
        for key in self.paths:
            #print(key, end=': ')
            for mNode in self.paths[key]:
                print(mNode,end=' ')
            #print("")
   

    # function to find all Ancestors
    def findAllAncestors(self):    
        # Mark all the vertices as not visited 
        visited =[False]*(self.V)
        for i in range(self.V):
            for j in range(self.V):
                visited[j] = False
            self.findAncestors(i,i,visited)
        for j in range(self.V):
            #print("Ancestors of: ", j)
            for n in self.ancestor[j]:
                #print(n, end = ' ')
                pass
            #print('')
            
    def findAncestors(self, u, v, visited):
        visited[v] = True
        self.ancestor[u].append(v)
        
        for adjNode in self.graph[v]:
            if (not visited[adjNode.v]) and adjNode.dir == IN:
                self.findAncestors(u, adjNode.v, visited)
        
    
    # function to add an edge to graph 
    def addEdge(self, u, v): 
        self.graph[u].append(v) 
    
    
    #print path
    def printPath(self, path, direction, length):
        
        #print('path: ', path)
        #print('direction', direction)
        #print('length:', length)

        for i in range(length-1):
            #print(path[i], end=' ')
            pass
            #if direction[i] == OUT:
                #print('->', end = ' ')
            #else: #print('<-', end = ' ')
        #print(path[length-1])
        

    def savePath(self, path, direction, length):
        self.nodeCnt += 1  # used for unique key generation
        nodeCnt = self.nodeCnt
        key = (nodeCnt, path[0], path[length-1])
        pathNodes = []
        for i in range(length-1):
            nodeId = path[i]
            #print(path[i], end=' ')
            if direction[i] == OUT:
                #print('->', end = ' ')
                arrow = '->'
            else: 
                #print('<-', end = ' ')
                arrow = '<-'
            mNode = PathNode(nodeId, arrow)
            pathNodes.append(mNode)
        nodeId = path[length-1]
        arrow = '|'
        mNode = PathNode(nodeId, arrow)
        pathNodes.append(mNode)
        #print(path[length-1])
        self.paths[key] = pathNodes
        #print("Info: Saved",key)
        
        
    # Prints all paths from 's' to 'd' 
    def printAllPaths(self, s, d):
        visited =[False]*(self.V)
        path = [0]*(self.V+1)
        self.path_index = 0
        direction = [False]*(self.V)
        
        self.printAllPathsUtil(s, d, visited, path, direction, self.path_index)
    
    
    
    '''A recursive function to print all paths from 'u' to 'd'. 
    visited[] keeps track of vertices in current path. 
    path[] stores actual vertices and path_index is current 
    index in path[]'''
    def printAllPathsUtil(self, u, d, visited, path, direction, path_indexX): 
  
        # Mark the current node as visited and store in path 
        visited[u]= True
        path[self.path_index] = u 
        self.path_index += 1
  
        # If current vertex is same as destination, then print 
        # current path[] 
        if u == d: 
            if direction[self.path_index-2] == OUT:
                self.printPath(path, direction, self.path_index) 
            else: 
                # find if d has a loop
                # Recur for all the vertices adjacent to this vertex 
                flag = False
                
                for adjNode in self.graph[d]:
                    if adjNode.v == d:
                        path[self.path_index] = d
                        direction[self.path_index-1] = OUT
                        self.path_index += 1
                        flag = True
                        break
                if flag:
                    self.printPath(path, direction, self.path_index)
                    self.path_index -= 1
        # // If current vertex is not destination
        else:
            for adjNode in self.graph[u]:
                if (not visited[adjNode.v]):
                    if adjNode.dir == OUT:
                        direction[self.path_index-1] = OUT
                    else:
                        direction[self.path_index-1] = IN
                    self.printAllPathsUtil(adjNode.v, d, visited, path, direction, self.path_index)
        # Remove current vertex from path[] and mark it as unvisited 
        self.path_index -= 1 
        visited[u]= False
    
    
    def findConnectingRoute(self, a, b, C):
        
        # Mark all the vertices as not visited 
        visited =[False]*(self.V)
        path = [0]*(self.V+1)
        self.path_index = 0
        direction = [False]*(self.V)
        
        # Call the recursive helper function to print all paths 
        self.findConnectingRouteUtil(a, b, C, visited, path, direction, self.path_index)
        
        
    def findConnectingRouteUtil(self, u, d, C, visited, path, direction, path_indexX):
    
        # Mark the current node as visited and store in path 
        visited[u]= True
        path[self.path_index] = u 
        self.path_index += 1
        
        # If current vertex is same as destination, then print 
        # current path[] 
        if u == d: 
            if direction[self.path_index-2] == OUT:
                if self.isOpen(path, direction, self.path_index, C):
                    #print("Debug: at line",207)
                    self.printPath(path, direction, self.path_index) 
                    self.savePath(path, direction, self.path_index) 
            # if d has a loop
            else: 
                flag = False
                
                for adjNode in self.graph[d]:
                    if adjNode.v == d:
                        path[self.path_index] = d
                        direction[self.path_index-1] = OUT
                        self.path_index += 1
                        flag = True
                        break
                if flag:
                    if self.isOpen(path, direction, self.path_index, C):
                        self.printPath(path, direction, self.path_index)
                        #print("Debug: at line",224)
                        self.savePath(path, direction, self.path_index) 
                    self.path_index -= 1
        # // If current vertex is not destination
        else:
            for adjNode in self.graph[u]:
                if (not visited[adjNode.v]):
                    if adjNode.dir == OUT:
                        direction[self.path_index-1] = OUT
                    else:
                        direction[self.path_index-1] = IN
                    self.findConnectingRouteUtil(adjNode.v, d, C, visited, path, direction, self.path_index)
                    #self.printAllPathsUtil(adjNode.v, d, visited, path, direction, self.path_index)
        # Remove current vertex from path[] and mark it as unvisited 
        self.path_index -= 1 
        visited[u]= False
    
    
    def isOpen(self, path, direction, length, C):
        return self.isOpenUtil(path, direction, length, C, 0)
    
    
    def isOpenUtil(self, path, direction, length, C, v):
        if v == length-1:
            return True
            
        if direction[v] == OUT:
            # v is the second last node
            if (v==length-2):
                return True
                
            if direction[v+1] == OUT:
                # path[v+1] is not in C
                flag = True
                for i in C:
                    if path[v+1] == i:
                        flag =False
                        break
                if flag and self.isOpenUtil(path, direction, length, C, v+1):
                    return True
                    
            elif direction[v+1]==IN:
                # path[v+1] is in An(C)
                flag = False
                for i in C:
                    for j in self.ancestor[i]:
                        if path[v+1] == j:
                            #flag = True
                            #break
                            continue   # for dynamic data
                            
                    if flag: break
                if flag and self.isOpenUtil(path, direction, length, C, v+1):
                    return True
                    
        else:
            # path[v+1] is not in C
            flag = True
            for i in C:
                if path[v+1] == i:
                    flag = False
                    break
            if flag and self.isOpenUtil(path, direction, length, C, v+1):
                return True
    
        return False


    def isExistOpenRoute(self, s, d, C):
        self.findConnectingRoute(s, d, C)
        cntOpenPahts = 0
        # count saved paths with (*, s, d) as key
        for key in self.paths:
            if key[1]==s and key[2]==d:
                cntOpenPahts += 1
        if cntOpenPahts > 0: return True
        else: return False
   

    
def open_route(n, graph, p, C_):
    #print('C: ', C_)
    #print(graph)
    g = Graph(n) 
    s = p[0] 
    d = p[1]
    
    for i in range(n):
        for j in range(n):
            if graph.iloc[i,j] == 1:
                #print(graph.iloc[i,j])
                if i == j: 
                    g.addEdge(i, AdjNode(j, OUT))
                else:
                    g.addEdge(i, AdjNode(j, OUT))
                    g.addEdge(j, AdjNode(i, IN))
            #g.printAllPaths(i, j)
    #print("Following are all different paths from % d to % d :" %(s, d)) 
    #g.printAllPaths(s, d) 
    g.findAllAncestors()

    C = C_
    open_path = g.isExistOpenRoute(s,d,C)
    return open_path


    