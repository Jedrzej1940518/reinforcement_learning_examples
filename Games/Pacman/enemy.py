
import copy
import string
from common import *

class Mode(IntEnum):
    SCATTER = 0,
    CHASE = 1,
    FRIGHTENED = 2
        
class Ghost:    
        
    def __init__(self, name: string, pacman):
        
        self.name = name
        self.pos = find_color_position(name_to_colors[name])
        
        self.phase_level = 0
        self.scatter_frames = 0
        self.chase_frames = 0        
        self.waited = False
        self.mode = Mode.CHASE
        self.direction = Direction.LEFT
        self.pacman = pacman
        self.target_tile = pacman.pos
    
    def determine_direction(self):

        direction = self.direction
        #cant choose opposite direction
        filtered_directions = [d for d in directions if d != oppoisite_direction[self.direction] and d!= Direction.NONE]
        
        #get closer to target
        if intersection_on_grid(self.pos):
            possible_dirs = [d for d in filtered_directions if not collision_on_grid(next_position(d, self.pos))]
            distances = [[d, distance_to_tile(next_position(d, self.pos), self.target_tile)] for d in possible_dirs]
            direction = min(distances, key=lambda x: x[1])[0]
        
        next_pos = next_position(direction, self.pos)

        #avoid collisions
        while collision_on_grid(next_pos):
            direction +=1
            direction %= Direction.NONE
            if direction == oppoisite_direction[self.direction]:
                continue
            next_pos = next_position(direction, self.pos)
                
        self.direction = direction
        pass
    
    def update(self):
        
        self.target_tile = self.pacman.pos

        if self.mode == Mode.FRIGHTENED:
            self.waited = not self.waited
            #in frigthened mode we skip a frame because ghosts are "slower"
            if not self.waited:
                return
        
        self.determine_direction()

        self.pos = next_position(self.direction, self.pos)
        self.pos.x %= exported_grid.ncols
        self.pos.y %= exported_grid.nrows
        
        if self.pos == self.pacman.pos:
            self.pacman.kill()
        
    def draw(self):
        color = name_to_colors[self.name]
        draw_character(self.pos, color)
