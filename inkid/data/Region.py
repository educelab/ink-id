import random


class Region:
    def __init__(self, region_id, ppm, bounds=None):
        self._region_id = region_id
        self.ppm = ppm

        if bounds is not None:
            self._bounds = bounds
        else:
            self._bounds = self.ppm.get_default_bounds()

    def get_points(self, restrict_to_surface,
                   grid_spacing=None,
                   probability_of_selection=None):
        if probability_of_selection is None:
            probability_of_selection = 1.0
        if grid_spacing is None:
            grid_spacing = 1

        points = []
        x0, y0, x1, y1 = self._bounds
        for y in range(y0, y1, grid_spacing):
            for x in range(x0, x1, grid_spacing):
                if random.random() > probability_of_selection:
                    continue
                if restrict_to_surface:
                    if self.ppm.is_on_surface(x, y):
                        points.append([self._region_id, x, y])
        return points
