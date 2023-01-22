from PIL import Image


def create_gif(epoch, len_t):
    files = []
    for i in range(len_t):
        seq = str(i)
        file_names = 'results2Dnew/epoch_' + str(epoch) + "/t_" + seq + '.png'
        files.append(file_names)

    # Create the frames
    frames = []
    for i in files:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(f'results2Dnew/animation_{epoch}.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=40, loop=0)
