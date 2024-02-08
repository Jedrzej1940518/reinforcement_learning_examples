
import copy
from common import *



class Pacman:
    global grid, consumed_grid
    
    def __init__(self):
        self.pos = find_color_position('yellow')
        self.direction = Direction.RIGHT
        self.pill = False
        self.isDead = False
        self.score = 0

    def update(self):

        next_pos = next_position(self.direction, self.pos)
        
        next_pos.x %= exported_grid.ncols
        next_pos.y %= exported_grid.nrows
        
        #collision with wall
        if collision_on_grid(next_pos):
            next_pos = self.pos
        #consuming food
        if food_on_grid(next_pos):
            consumed_grid[next_pos.y, next_pos.x] = color_to_value['white']
            self.score +=1
        #eating pills
        if pill_on_grid(next_pos):
            consumed_grid[next_pos.y, next_pos.x] = color_to_value['white']
            self.pill = True
        #colliding with enemy
        if enemy_on_grid(next_pos):
            self.isDead = True
            
        self.pos = next_pos

    def set_direction(self, dir: Direction):
        self.direction = dir

    def draw(self):
        color = "orange" if self.pill == True else "yellow"
        draw_character(self.pos, color)

    def kill(self):
        self.isDead = True 