import ipyleaflet as ipl


class BaseMap:
    def __init__(self):
        self._m = ipl.Map(layers=[], crs=ipl.projections.Simple)
