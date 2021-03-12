import datetime
import os

import iris
import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource,
                          Slider, LabelSet,
                          LinearColorMapper,
                          ColorBar, GeoJSONDataSource)
from bokeh.palettes import brewer
from bokeh.plotting import figure


def read_data_df():
    filepath = os.path.join(os.path.dirname(__file__), 'data/1880-2020.csv')
    print(filepath)
    df = pd.read_csv(filepath, header=4)
    years = [int(str(val)[:4]) for val in df['Year'].values]
    months = [int(str(val)[4:]) for val in df['Year'].values]
    ddf = {'Year': years, 'Month': months, 'Values': df['Value'].values}
    ddf = pd.DataFrame(ddf, columns=['Year', 'Month', 'Values'])
    return ddf


def expand_df(ddf, year_start=1880, year_end=2020):
    nyears = year_end - year_start + 1

    month_angles = [(m + 1) * 360 / 12. for ny in np.arange(0, nyears) for m in range(12)]
    year_angles = [month_angle - (ny + 1) * 30. / nyears for ny in np.arange(0, nyears)
                   for month_angle in month_angles[:12]]

    tradius = 0.6
    tx = [tradius * np.sin((month_angle - 15) * np.pi / 180.) for month_angle in month_angles[:12]]
    ty = [tradius * np.cos((month_angle - 15) * np.pi / 180.) for month_angle in month_angles[:12]]
    print(len(year_angles))
    pdd = ddf.loc[(ddf['Year'].between(year_start, year_end, inclusive=True))]
    pdd.insert(3, "Month_angles", month_angles)
    pdd.insert(4, "Year_angles_start", year_angles)
    pdd.insert(5, "Year_angles_end", year_angles)
    pdd.insert(6, "End_line", 2 + pdd['Values'].values)

    # create new column for color of the bars
    pdd['Colors'] = pdd.Values.apply(lambda x: 'lightcoral' if x >= 0. else 'steelblue')
    return pdd


def create_base_plot(source, width=600, height=600, year_start=1880, year_end=2020):
    label_angles = [75 - m * 30 for m in range(12)]
    month_names = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    # create chart
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    p = figure(plot_width=width, plot_height=height, title="",
               x_axis_type=None, y_axis_type=None,
               x_range=(-4, 4), y_range=(-4, 4),
               min_border=0, outline_line_color="black",
               background_fill_color="#f0e1d2", tools=TOOLS)

    month_angles_df = pdd.loc[(pdd['Year'] == year_start) & (ddf['Month'].between(1, 12, inclusive=True))]
    '''
    hover = p.select(dict(type=HoverTool))
    p.hover.tooltips = [
        ('Year', '@Year'),
        ('Month', '@Month')
    ]
    '''

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # circular axes and lables
    labels = np.arange(-2, 2, 0.5)
    radii = np.arange(0., 4, 0.5)
    p.circle(0, 0, radius=radii, fill_color=None, line_color="white")
    p.text(0, radii[1:], [str(r) for r in labels[1:]],
           text_font_size="11px", text_align="center", text_baseline="middle")

    for m, month_angle in enumerate(month_angles_df['Month_angles'].values):
        # radial axes for 12 months
        p.annular_wedge(0, 0, 0.5, 4., month_angle, month_angle, color="white",
                        start_angle_units='deg', end_angle_units='deg')

        tradius = 0.6
        tx = tradius * np.sin((month_angle - 15) * np.pi / 180.)
        ty = tradius * np.cos((month_angle - 15) * np.pi / 180.)

        label_angle = month_angle + 45 * (m + 1)
        print(month_angle, label_angle)
        p.text(tx * 1.2, ty * 1.2, [str(month_names[m])], angle=(label_angles[m]) * np.pi / 180.,
               text_font_size="14px", text_align="center", text_baseline="middle")

    p.annular_wedge(0, 0, 2, 'End_line', 'Year_angles_start', 'Year_angles_end',
                    color='Colors', source=source, start_angle_units="deg",
                    end_angle_units="deg", fill_alpha=0.5, line_alpha=0.5)
    return p


def read_extract_yearly_mean_data(year=2020):
    ncfile_path = os.path.join(os.path.dirname(__file__), 'data/air.mon.anom.nc')
    temp = iris.load_cube(ncfile_path)
    temp = temp.intersection(longitude=(-180, 180))

    # time is set as a Aux coordinate - wrong!
    time = temp.coord('time')
    # remove time bounds - cos it is wrong
    time.bounds = None
    time.guess_bounds()
    # promoting aux coord to dim coordinate
    iris.util.promote_aux_coord_to_dim_coord(temp, 'time')

    # get the indices of all months
    xinds = [time.units.date2num(datetime.datetime(year, mn, 1)) for mn in range(1, 13)]
    data_ind = np.array([np.where(np.array(time.points) == xind)[0][0] for xind in xinds])

    annual_mean = temp[data_ind].collapsed('time', iris.analysis.MEAN)
    # print(annual_mean)
    return annual_mean


def create_map_plot(map_source):
    if map_source:
        colors = brewer['RdBu'][9]
        with open(os.path.join(os.path.dirname(__file__), 'data/countries.geo.json'), "r") as f:
            countries = GeoJSONDataSource(geojson=f.read())

        cube_data = map_source.data['cube_data'][0]
        lons = map_source.data['lons'][0]
        lats = map_source.data['lats'][0]
        # Set up plot
        x_range = (-180, 180)  # could be anything - e.g.(0,1)
        y_range = (-90, 90)

        # plot = figure(plot_height=49, plot_width=360, x_range=x_range, y_range=y_range)
        # plot = figure(plot_height=300, plot_width=600, x_range=x_range, y_range=y_range)
        mplot = figure(plot_height=600, plot_width=1200, x_range=x_range, y_range=y_range,
                       tools=["pan, reset, save, wheel_zoom, hover"],
                       title='Annual Mean Temperature anomaly (C)',
                       x_axis_label='Longitude', y_axis_label='Latitude')#, aspect_ratio=2)
        color_mapper_z = LinearColorMapper(palette=colors, low=-3, high=3)
        mplot.image(image='cube_data', x=min(lons), y=min(lats), dw=max(lons) - min(lons),
                    dh=max(lats) - min(lats), source=map_source,
                    color_mapper=color_mapper_z)

        # add titles
        mplot.text(0, -60, 'label', source=map_source, text_color="firebrick",
                   text_align="center", text_font_size="35pt", text_alpha=0.5)

        color_bar = ColorBar(color_mapper=color_mapper_z, location=(0, 0))

        mplot.patches("xs", "ys", color=None, line_color="grey", source=countries, alpha=0.5)
        mplot.add_layout(color_bar, 'right')
        return mplot


def update_monthly_data(attrname, old, new):
    new_pdd = pdd.copy(deep=True)
    # create new column for color of the bars on selection
    new_pdd.loc[(new_pdd['Year'] == new) & (new_pdd['Values'] >= 0), 'Colors'] = 'maroon'
    new_pdd.loc[(new_pdd['Year'] == new) & (new_pdd['Values'] < 0), 'Colors'] = 'midnightblue'
    new_pdd.loc[(new_pdd['Year'] == old) & (new_pdd['Values'] >= 0), 'Colors'] = 'lightcoral'
    new_pdd.loc[(new_pdd['Year'] == old) & (new_pdd['Values'] < 0), 'Colors'] = 'steelblue'

    # make the seclected wedge bit wider
    angle_dummy_new = new_pdd.loc[(pdd['Year'] == new), 'Year_angles_start']
    angle_dummy_old = new_pdd.loc[(pdd['Year'] == old), 'Year_angles_start']

    new_pdd.loc[(pdd['Year'] == new), 'Year_angles_start'] = angle_dummy_new - 1
    new_pdd.loc[(pdd['Year'] == new), 'Year_angles_end'] = angle_dummy_new + 1

    # pdd.loc[(pdd['Year'] == old), 'Year_angles_start'] = angle_dummy_new
    # pdd.loc[(pdd['Year'] == old), 'Year_angles_end'] = angle_dummy_new
    source.data = new_pdd


def update_map_data(attrname, old, new):
    print('in update_map_data', new, old)
    cube = read_extract_yearly_mean_data(year=int(new))
    title = ['Annual mean temperature %s' % (new)]
    map_source.data['cube_data'] = [cube.data]
    map_source.data['label'] = [new]
    #map_plot.title.text = [title]



ddf = read_data_df()
year_start = 1880
year_end = 2020

# Time series data
pdd = expand_df(ddf, year_start=year_start, year_end=year_end)
source = ColumnDataSource(data=pdd)

# Set up widgets
slider_year = Slider(title="Year", value=year_end, start=year_start, end=year_end, step=1)
print(int(slider_year.value))
# Generate contours from 2d cube


cube = read_extract_yearly_mean_data(year=int(slider_year.value))
lons = cube.coord('longitude').points
lats = cube.coord('latitude').points

map_source = ColumnDataSource(data={'cube_data': [cube.data], 'lons': [lons], 'lats': [lats],
                                    'label': [slider_year.value]})

# Generate a base plot of axes etc
plot = create_base_plot(source, year_start=year_start, year_end=year_end)
map_plot = create_map_plot(map_source)

slider_year.on_change('value', update_monthly_data)
slider_year.on_change('value', update_map_data)

inputs = column(slider_year)
curdoc().add_root(row(plot, column(inputs, map_plot), width=1000))
curdoc().title = "Sliders"
# show(row(inputs, p, width=800))
