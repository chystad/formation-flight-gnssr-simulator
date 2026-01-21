from object_definitions.TwoLineElement_def import TLE

class ClassicalOrbitalElements:
    def __init__(self, tle: TLE) -> None:
        """
        =========================================================================================================
        ATTRIBUTES (Same as Basilisk's orbitalMotion.ClassicElements):
            a           Semi-major axis (avg distance from the Earth's center)
            e           Eccentricity
            i           Inclination
            Omega       Ascending node / right ascension of the ascending node (RAAN) / Longitude of ascending node
            omega       Argument of periapsis
            f           True anomaly angle
            rmag        Position vector magnitude
            alpha       Specific orbital energy
            rPeriap     Smallest distance between the primary and secondary 
            rApoap      Largest distance between the primary and secondary
        =========================================================================================================
        """
        
        self.a = None # 

        self.e = tle.eccentricity
        self.i = tle.inclination 
        self.Omega = tle.raan
        self.omega = None
        self.f = None # TODO


        
        pass