class SolarPanel:
    def __init__(self,
                 nHat_B: list[int],
                 panel_area: float,
                 panel_efficiency: float) -> None:
        """
        Define and store attributes required by the Basilisk SimpleSolarPanel object 
        from config parameters (parsed from Config)

        NOTE: All parsing and input verifications are performed by the Config method:
            'generate_solar_panel_instances_from_config'.
        All input parameters will therefore be assumed to be correct.

        params:
            nHat_B (list[int]): Panel normal unit vec in body (B)
        """
        self.nHat_B = nHat_B
        self.panel_area = panel_area
        self.panel_efficiency = panel_efficiency


    # def calculate_face_with_largest_panel_area(self, all_solar_panels: list[SolarPanel]) -> list[int]:
    #     """
    #     From 
    #     """
        
    #     # TODO

    #     r_PB_B = [0, 0, 0]
        
    #     return r_PB_B