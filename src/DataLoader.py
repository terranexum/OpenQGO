import json
import os

import pandas
import geopandas

#stack
from collections import deque

class QGOFile: 

     file_path = ""

     fields = []
     values = []

     encapsulating_marks = {
          
          '{' : '}',
          '[' : ']',
          '(' : ')',
          '<' : '>',
          "\"" : "\""
     }

     separating_marks = [

          ',',
          ':',
          

     ]

     orig_file = None
     geojson_file = None 

     
     def prepareFile(self):
          

          encapsulating_start_marks = list(self.encapsulating_marks.keys())
          encapsulating_end_marks = list(self.encapsulating_marks.values())

          i = 0

          encapsulating_stack = deque()
          encapsulating_indices = []

          #check each line of data file
          for line in self.orig_file:
               
               
               if i == 1: break

               #create stack to store encapsulating indices
               

               index = 0

               #iterate over characters
               for char in line:
                    
                    #check if character starts an encapsulating sequence
                    if char in encapsulating_start_marks:
                         
                         

                         if char == "\"" and encapsulating_stack[-1][0] == "\"":
                              print('here')
                              start_idx = encapsulating_stack[-1][1]

                              encapsulating_stack.pop()
                              encapsulating_indices.append((start_idx, index))

                         else:
                              encapsulating_stack.append((char, index))
                         
                    #if we've gotten to an end encapsulating mark
                    elif char in encapsulating_end_marks:
                         
                         #check last item we added in stack
                         last_item = encapsulating_stack[-1]

                         #if the end mark we found matches up with the starting mark, we can take it out
                         if self.encapsulating_marks[last_item[0]] == char:

                              encapsulating_stack.pop()
                              encapsulating_indices.append((last_item[1], index))

                    index += 1

               i += 1

          for indices in encapsulating_indices:

               print(indices)
               



     def convertTxtToGeoJSON(self):

          fieldValueDict = {}

          for line in self.orig_file:
               

               split : list(str) = line.strip().split()
               print(split)
          
          pass

     def convertJSONToGeoJSON():
          
          pass
     

     #def multiSplit(a):

# def removeUnnecesssaryMarks():




     

     def __init__(self, file_path : str) -> None:

               self.file_path = file_path
               self.orig_file = open(self.file_path, "r")

               #self.fields = fields

               name , ext = os.path.splitext(file_path)
               
               #self.convertTxtToGeoJSON()
          #
          #     if ext == ".geojson":
          #          geojson_file = self.orig_file
          #     
          #     elif ext == ".txt":
          #          
          #          geojson_file = self.convertTxtToGeoJSON()

          #     elif ext == ".json":
          #          
          #          geojson_file = self.convertJSONToGeoJSON()

               pass
     

qgoFile = QGOFile("C:\\Users\\Anish J\\Desktop\OpenQGO\src\\ELEC.txt")

qgoFile.prepareFile()