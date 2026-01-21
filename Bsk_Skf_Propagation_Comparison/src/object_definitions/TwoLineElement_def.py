from datetime import datetime, timedelta

class TLE:
    """
    Parse Two-Line Element (TLE) strings using fixed-column indices.
    Usage:
        tle = TLE(tle_line_1, tle_line_2)
        print(tle.epoch)  # datetime (UTC)
        print(tle.meanMotion)  # rev/day
    """

    def __init__(self, tle_line_1: str, tle_line_2: str):
        """
        =========================================================================================================
        ATTRIBUTES:
            satCatalogNum           
            classification          
            launchYear              
            launchNumOfYear         
            launchPiece             
            epoch                   
            firstDerMeanMotion      
            secondDerMeanMotion     
            bStar                   
            ephemerisType           
            elemSetNum              
            inclination             
            raan                    
            eccentricity            
            argOfPerigee            
            meanAnomaly             
            meanMotion              
            revNoAtEpoch            
        =========================================================================================================
        """
        # store raw
        self.line1 = tle_line_1.rstrip('\n')
        self.line2 = tle_line_2.rstrip('\n')

        # Right-pad to avoid IndexError if trailing spaces are missing
        l1 = self.line1.ljust(69)
        l2 = self.line2.ljust(69)

        # ---- Line 1 parsing (1-indexed cols in comments) ----
        # 1   line number '1'
        self.satCatalogNum      = l1[2:7].strip()            # cols 3-7
        self.classification     = l1[7:8].strip()            # col 8
        self.launchYear         = l1[9:11].strip()           # cols 10-11
        self.launchNumOfYear    = l1[11:14].strip()          # cols 12-14
        self.launchPiece        = l1[14:17].strip()          # cols 15-17
        epoch_year2             = l1[18:20].strip()          # cols 19-20
        epoch_day               = l1[20:32].strip()          # cols 21-32
        first_der_str           = l1[33:43].strip()          # cols 34-43
        second_der_field        = l1[44:52]                  # cols 45-52 (compact exponent)
        bstar_field             = l1[53:61]                  # cols 54-61 (compact exponent)
        self.ephemerisType      = l1[62:63].strip()          # col 63
        self.elemSetNum         = l1[64:68].strip()          # cols 65-68
        # checksum1             = l1[68]                     # col 69

        # conversions
        self.firstDerMeanMotion = float(first_der_str.replace(' ', '')) if first_der_str else 0.0
        self.secondDerMeanMotion = self._parse_implied_exponent(second_der_field)
        self.bStar               = self._parse_implied_exponent(bstar_field)
        self.epoch               = self._parse_epoch(epoch_year2, epoch_day)  # datetime (UTC)

        # ---- Line 2 parsing ----
        # 1   line number '2'
        satCatalogNum2          = l2[2:7].strip()            # cols 3-7
        self.inclination        = float(l2[8:16].strip())    # cols 9-16 (deg)
        self.raan               = float(l2[17:25].strip())   # cols 18-25 (deg)
        ecc_str                 = l2[26:33].strip()          # cols 27-33 (no decimal point)
        self.argOfPerigee       = float(l2[34:42].strip())   # cols 35-42 (deg)
        self.meanAnomaly        = float(l2[43:51].strip())   # cols 44-51 (deg)
        self.meanMotion         = float(l2[52:63].strip())   # cols 53-63 (rev/day)
        self.revNoAtEpoch       = l2[63:68].strip()          # cols 64-68
        # checksum2             = l2[68]                     # col 69

        # ecc has implied decimal
        self.eccentricity = float(f"0.{ecc_str}") if ecc_str else 0.0

        # simple consistency check
        if self.satCatalogNum and satCatalogNum2 and self.satCatalogNum != satCatalogNum2:
            raise ValueError(f"TLE catalog number mismatch: {self.satCatalogNum} vs {satCatalogNum2}")
        

    def __repr__(self) -> str:
        """
        Define the object's output format when 'print(TLE_instance)' is called
        """

        leftPadding = "    "

        printableString = f"""---------------------------- TLE attributes ----------------------------
{leftPadding} satCatalogNum:        {self.satCatalogNum}
{leftPadding} classification:       {self.classification}
{leftPadding} launchYear:           {self.launchYear}
{leftPadding} launchNumOfYear:      {self.launchNumOfYear}
{leftPadding} launchPiece:          {self.launchPiece}
{leftPadding} epoch:                {self.epoch}
{leftPadding} firstDerMeanMotion:   {self.firstDerMeanMotion}
{leftPadding} secondDerMeanMotion:  {self.secondDerMeanMotion}
{leftPadding} bStar:                {self.bStar}
{leftPadding} ephemerisType:        {self.ephemerisType}
{leftPadding} elemSetNum:           {self.elemSetNum}
{leftPadding} inclination:          {self.inclination}
{leftPadding} raan:                 {self.raan}
{leftPadding} eccentricity:         {self.eccentricity}
{leftPadding} argOfPerigee:         {self.argOfPerigee}
{leftPadding} meanAnomaly:          {self.meanAnomaly}
{leftPadding} meanMotion:           {self.meanMotion}
{leftPadding} revNoAtEpoch:         {self.revNoAtEpoch}
------------------------------------------------------------------------""" 

        return printableString


    @staticmethod
    def _parse_implied_exponent(field: str) -> float:
        """
        Convert TLE compact exponent fields like ' 00000-0', ' 12345-4', '-11606-4' to float.
        Format: [optional sign][5 digits][exp sign][exp digit(s)]
        Value = mantissa * 10^(exp - 5)
        """
        s = field.strip()
        if not s:
            return 0.0
        # Ensure there's at least mantissa + exp sign + exp digit
        if len(s) < 7:
            s = s.rjust(7)
        # Find last '+' or '-' that isn't the very first character
        pos = max(s.rfind('+', 1), s.rfind('-', 1))
        if pos == -1:
            mant_str, exp_str = s[:-1], '+' + s[-1]
        else:
            mant_str, exp_str = s[:pos], s[pos:]
        mant = int(mant_str)
        exp = int(exp_str)
        return mant * (10.0 ** (exp - 5))


    @staticmethod
    def _parse_epoch(year2: str, day_str: str) -> datetime:
        """
        Convert TLE epoch (YY, DDD.DDDDDDDD) to a UTC datetime.
        NORAD rule: 57-99 => 1900s, 00-56 => 2000s.
        """
        yy = int(year2)
        year = 1900 + yy if yy >= 57 else 2000 + yy
        day_of_year = float(day_str)
        doy_int = int(day_of_year)
        frac_day = day_of_year - doy_int
        dt0 = datetime(year, 1, 1) + timedelta(days=doy_int - 1, seconds=frac_day * 86400.0)
        return dt0
    

    # Optional convenience: ISO string for epoch
    @property
    def epoch_iso(self) -> str:
        return self.epoch.strftime("%Y-%m-%dT%H:%M:%S.%fZ").rstrip('0').rstrip('.')