import pygame,sys,pymunk
from pygame.constants import MOUSEBUTTONDOWN

def create_apple(space,pos):
    body = pymunk.Body(1,100,body_type=pymunk.Body.DYNAMIC)
    body.position = pos
    shape = pymunk.Circle(body, 80)
    space.add(body,shape)
    return shape

def draw_apples(apples):
    for apple in apples:
        pos_x = int(apple.body.position.x)
        pos_y = int(apple.body.position.y)
        pygame.draw.circle(screen,(0,0,0), (pos_x,pos_y),80)

def stactic_ball(space,pos):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = pos
    shape = pymunk.Circle(body, 40)
    space.add(body,shape)
    return shape

def draw_static_ball(balls):
    for ball in balls:
        pos_x = int(ball.body.position.x)
        pos_y = int(ball.body.position.y)
        pygame.draw.circle(screen,(0,0,0), (pos_x,pos_y), 40)



pygame.init()
screen = pygame.display.set_mode((800,800)) #creating the display surface
clock = pygame.time.Clock() #creating the game clock
space = pymunk.Space()
space.gravity = (0,500)
apples = []


balls = []
balls.append(stactic_ball(space,(500,500)))
balls.append(stactic_ball(space,(250,600)))

while True: # Game loop
    for event in pygame.event.get(): #checking for user input
        if event.type == pygame.QUIT: #input to close the simulation
            pygame.quit()
            sys.quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            apples.append(create_apple(space,event.pos))
    screen.fill((217,217,217)) #background 
    draw_apples(apples)
    draw_static_ball(balls)
    space.step(1/50)
    pygame.display.update()
    clock.tick(120) #Limiting the frames per second to 120
    
