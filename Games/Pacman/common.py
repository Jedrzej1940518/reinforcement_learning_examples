from math import sqrt
import copy
from dataclasses import dataclass
from enum import IntEnum
import string
import numpy as np
import pygame
import exported_grid

# Screen dimensions
tile_size = 20

screen_width = tile_size*exported_grid.ncols
screen_height = tile_size*exported_grid.nrows

screen = pygame.display.set_mode((screen_width, screen_height))

grid = np.array(exported_grid.color_grid)
text_grid= np.array(exported_grid.text_grid)

color_to_value = {color: index for index, color in enumerate(exported_grid.colors)}
value_to_color = {index: color for index, color in enumerate(exported_grid.colors)}

consumed_grid = grid.copy()
score = 0

color_map = {
    #empty space
    'white': (255, 255, 255),
    #wall
    'black': (0, 0, 0),
    #food tile
    'green': (0, 255, 0),
    #pill
    'blue': (0, 0, 255),
    #pacman
    'yellow': (255, 255, 0),
    #pacman on drugs
    'orange' : (255,165,0),
    
    #enemies
    'red': (255, 0, 0),
    'pink' : (255, 192, 203),
    'purple': (128, 0, 128),
    'teal': (0, 128, 128)
}


# Pac-Man settings

@dataclass()  
class Position:
    x: int
    y: int
    
class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    NONE = 4
    DIRECTION_NUM = 5
    
directions = [direction for direction in range(Direction.DIRECTION_NUM)]
oppoisite_direction = {Direction.UP : Direction.DOWN, 
                       Direction.DOWN : Direction.UP, 
                       Direction.LEFT : Direction.RIGHT, 
                       Direction.RIGHT : Direction.LEFT,
                       Direction.NONE : Direction.NONE,
                       Direction.DIRECTION_NUM : Direction.NONE} 

name_to_colors =  {'blinky': 'red', 'pinky': 'pink', 'inky':'purple', 'clyde': 'teal'}

def next_position(dir: Direction, position: Position) -> Position:
    pos = copy.copy(position)
    if dir == Direction.UP:
        pos.y -=1
    elif dir == Direction.DOWN:
        pos.y +=1
    elif dir == Direction.LEFT:
        pos.x -=1
    elif dir == Direction.RIGHT:
        pos.x +=1
    return pos
    

def draw_character(pos: Position, color):
    rect = pygame.Rect(pos.x * tile_size, pos.y * tile_size, tile_size, tile_size)
    pygame.draw.rect(screen, color_map[color], rect)
    
           
def find_color_position(color: string) -> Position:
    for y in range(exported_grid.nrows):
        for x in range(exported_grid.ncols):
            if grid[y,x] == color_to_value[color]:
                print("color pos x y ", color, x,y)
                return Position(x,y)

def enemy_colors():
    return ['red', 'pink', 'purple', 'teal']

def enemy_values():
    return [color_to_value[color] for color in enemy_colors()]

def pill_on_grid(pos: Position) -> bool:
    return consumed_grid[pos.y, pos.x] == color_to_value['blue']
def collision_on_grid(pos: Position) -> bool:
    return consumed_grid[pos.y, pos.x] == color_to_value['black']
def food_on_grid(pos: Position) -> bool:
    return consumed_grid[pos.y, pos.x] == color_to_value['green']
def pacman_on_grid(pos: Position) -> bool:
    return consumed_grid[pos.y, pos.x] == color_to_value['yellow']
def high_pacman_on_grid(pos: Position) -> bool:
    return consumed_grid[pos.y, pos.x] == color_to_value['orange']
def enemy_on_grid(pos: Position) -> bool:
    return consumed_grid[pos.y, pos.x] in enemy_values()

#todo fix A intersection
def intersection_on_grid(pos:Position) -> bool:
    return text_grid[pos.y, pos.x] == 'B' or text_grid[pos.y, pos.x] == 'A'

def distance_to_tile(origin: Position, to: Position) -> float:
    return sqrt((to.x - origin.x) ** 2 + (to.y - origin.y) ** 2)

def find_dir_to_tile(origin: Position, to: Position) -> Direction:
    
    if to.x > origin.x and not collision_on_grid(origin.x + 1, origin.y):
        return Direction.RIGHT
    elif to.x < origin.x and not collision_on_grid(origin.x - 1, origin.y):
        return Direction.LEFT
    elif to.y > origin.y and not collision_on_grid(origin.x, origin.y + 1):
        return Direction.DOWN
    elif to.y < origin.y and not collision_on_grid(origin.x, origin.y - 1):
        return Direction.UP
    
    elif not collision_on_grid(origin.x + 1, origin.y):
        return Direction.RIGHT
    elif not collision_on_grid(origin.x - 1, origin.y):
        return Direction.LEFT
    elif not collision_on_grid(origin.x, origin.y + 1):
        return Direction.DOWN
    elif not collision_on_grid(origin.x, origin.y - 1):
        return Direction.UP
    
    return Direction.RIGHT