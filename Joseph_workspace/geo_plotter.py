from DataPipeline import DataPipeline
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GeoPlotter():

    def __init__(self, df=None, merge=False):
        # if df is None:
        #     self.df = DataPipeline().clean(["name_change"])
        if 'ISO_A3' not in df.columns:
            raise ValueError("This dataframe does not have an ISO_A3 column. Please use the DataPipeline change_name transform to prepare it for plotting")
        else: 
            self.df = df
        world = gpd.read_file('worldmap.gpkg')
        self.merged = world.merge(self.df, left_on='ISO_A3', right_on='ISO_A3')


    def plot(self, col_name, year=None, continent='all', country='all'):
        fig, ax = plt.subplots(1, figsize=(10, 4))

        if year is None:
            df = self.merged
        else:
            df = self.merged[self.merged['Year'].astype(str) == year]
        df.plot(column=col_name, ax=ax, legend=True)
        ax.set_title(col_name)
        plt.show()
    
    def animate(self, col_name, video_name):
        fig, ax = plt.subplots(1, figsize=(10, 7))

        def update(year):
            df = self.merged[self.merged['Year'].astype(str) == str(year)]
            df.plot(column=col_name, ax=ax)
            ax.set_title(col_name + "\n" + str(year))

        animation.writer = animation.writers['ffmpeg']
        ani = animation.FuncAnimation(fig, update, 
            frames=sorted(set(self.df['Year'])),
            interval=self.df['Year'].max() - self.df['Year'].min() / 10)
        ani.save(video_name)
