class GroundStation:
    def __init__(self,
                 gs_tag: str,
                 latitude: float,
                 longitude: float,
                 altitude: float,
                 min_elev: float,
                 max_range: float | int) -> None:
        """
        Define and store attributes required by the Basilisk GroundLocation object 
        from config parameters (parsed from Config)

        NOTE: All parsing and input verifications are performed by the Config method:
            'generate_ground_station_instances_from_config'.
        All input parameters will therefore be assumed to be correct.

        params:
            gs_tag (str): Unique ground station tag
            latitude (float): Ground stattion position latitude [deg]
            longitude (float): Ground station position longitude [deg]
            altitude (float): Ground station position altiude [m]
            min_elev (float): Ground stattion minimum horizon elevation for contact [deg]
            max_range (float | int): Maximum distance between gs-sc for contact [m] (-1 corresponds to infinite range)
        """
        
        self.gs_tag: str = gs_tag
        self.lat: float = latitude
        self.long: float = longitude
        self.alt: float = altitude
        self.min_elev: float = min_elev
        self.max_range: float | int = max_range