import logging
from pathlib import Path

# Global definition of TLE output directory 
FOLLOWER_TLE_OUTPUT_DIR = Path("Bsk_Skf_Propagation_Comparison/output_data/follower_tle_files")

class TLE:
    """
    Contains methods to help parse, process and modify TLE files
    """

    def __init__(self,
                 satellite_param_path: str,
                 leader_tle_series_path: str,
                 inplane_separation_ang: float,
                 num_satellites: int) -> None:
        
        # Initialize flag
        verified_leader_TLE_series_file: bool = False

        # Set instance attributes
        self.satellite_param_path = satellite_param_path
        self.leader_tle_series_path = leader_tle_series_path
        self.inplane_separation_ang = inplane_separation_ang
        self.num_satellites = num_satellites
        self.verified_leader_TLE_series_file = verified_leader_TLE_series_file
        pass


    def generate_follower_tle_files(self) -> list[str]:
        """
        Generate TLE series files for follower satellites by copying the leader TLE series
        and shifting mean anomaly by i*self.inplane_separation_ang for follower i (1-indexed).
        The method also performs verification of any existing TLE files

        Output files:
            FOLLOWER_TLE_OUTPUT_PATH / f"follower_{i}_tle_series.txt"

        Returns:
            A list containing the full path to each of the follower tle file. 
            Type: (str) to be compatible with Skyfield 'Load.tle_file' method
        """
        verified_follower_tles = False # if True: follower TLE file(s) verified -> No need to generate new files
        out_follower_tle_paths: list[Path] = [] # Used for internal operations
        out_follower_tle_str_paths: list[str] = [] # Return variable
        num_followers = int(self.num_satellites) - 1

        # If no followers, don't generate files and return empty out_file_paths
        if num_followers <= 0:
            return []
        
        # Generate full follower TLE paths (str)
        for follower_i in range(1, num_followers+1):
            out_file_name = f"follower_{follower_i}_tle_series.txt"
            out_follower_tle_path = FOLLOWER_TLE_OUTPUT_DIR / out_file_name
            out_follower_tle_paths.append(out_follower_tle_path)
            out_follower_tle_str_paths.append(str(out_follower_tle_path))

        # Make directory if it doesn't already exist
        FOLLOWER_TLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Try to verify all existing follower TLE files, if they exist
        tle_verified = False
        for i, tle_path in enumerate(out_follower_tle_paths):
            tle_verified = self.verify_follower_TLE_series_file(tle_path)
            if not tle_verified:
                break
        if tle_verified:
            verified_follower_tles = True

        # If all follower TLE series files have been verified, simply return a list of all their paths
        if verified_follower_tles:
            logging.debug("No need to generate new follower TLE series files")
            return out_follower_tle_str_paths
        
        # Generate new TLE files for all followers
        else:
            sep_ang_deg = self.inplane_separation_ang
            leader_path = Path(self.leader_tle_series_path)
            
            # Read leader TLE series file
            with leader_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            # for each follower, generate a TLE file
            for follower_i in range(1, num_followers + 1):
                out_follower_tle_path = out_follower_tle_paths[follower_i-1]

                with out_follower_tle_path.open("w", encoding="utf-8") as out:
                    shift_deg = follower_i * sep_ang_deg

                    # Walk leader blocks
                    for b in range(0, len(lines), 3):
                        name_line = f"follower-{follower_i}\n"
                        line1 = lines[b + 1]
                        line2 = lines[b + 2]

                        # Parse mean anomaly, shift it, rewrite into fixed-width field, fix checksum
                        M_leader = self.parse_mean_anomaly_from_line2(line2)
                        M_follower = M_leader + shift_deg
                        line2_mod = self.set_mean_anomaly_and_fix_checksum(line2, M_follower)

                        # Write block
                        out.write(name_line)
                        out.write(line1 if line1.endswith("\n") else line1 + "\n")
                        out.write(line2_mod)

                logging.debug(f"Generated TLE file for 'follower-{follower_i}'")

        return out_follower_tle_str_paths
    

    ####################
    # Helper functions #
    ####################
    def verify_leader_TLE_series_file(self, leader_tle_path: Path) -> None:
        """
        Raise 'ValueError' if the leader TLE series file fails to fulfill any of the following criteria:
            * exists
            * correct format 
                - At least one 3-line block
                - All names are identical
                - line1 and line2 use TLE standard formatting
            * sorted in strictly ascending Epoch order
        """
        # Set exit flags
        exists = False
        correct_format = False
        correct_sorting = False   
        
        # Check if the file exists 
        if not leader_tle_path.exists():
            raise ValueError(f"Leader TLE series file does not exist at path: {leader_tle_path}")
        else:
            exists = True


        # --------------- Check leader TLE series format --------------- #
        # Read leader TLE
        with leader_tle_path.open("r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        if len(raw_lines) < 3:
            raise ValueError(
                f"Leader TLE series must contain at least one TLE (3 lines). Got {len(raw_lines)} lines."
            )

        if len(raw_lines) % 3 != 0:
            raise ValueError(
                f"Leader TLE series must be in 3-line blocks (name, line1, line2). "
                f"Got {len(raw_lines)} lines, not divisible by 3."
            )

        # Parse blocks and validate
        names: list[str] = []
        epoch_keys: list[tuple[int, float]] = []

        for b in range(0, len(raw_lines), 3):
            name = self._strip_newline(raw_lines[b]).strip()
            l1 = self._strip_newline(raw_lines[b + 1])
            l2 = self._strip_newline(raw_lines[b + 2])

            # Name must exist (non-empty)
            if not name:
                raise ValueError(f"Empty satellite name at block starting line {b+1}")

            names.append(name)

            # Basic "standard" checks
            if not self._is_tle_line1(l1):
                raise ValueError(f"Invalid TLE line1 at block starting line {b+1}: {l1!r}")
            if not self._is_tle_line2(l2):
                raise ValueError(f"Invalid TLE line2 at block starting line {b+1}: {l2!r}")

            # Length + checksum checks
            self._validate_checksum(l1, f"TLE line 1 (block starting line {b+1})")
            self._validate_checksum(l2, f"TLE line 2 (block starting line {b+1})")

            # Satellite number should match between line1 and line2
            sat1 = self._satnum_from_line(l1)
            sat2 = self._satnum_from_line(l2)
            if not sat1 or not sat2 or sat1 != sat2:
                raise ValueError(
                    f"Satellite number mismatch between line1 and line2 at block starting line {b+1}: "
                    f"line1 satnum='{sat1}', line2 satnum='{sat2}'."
                )

            # Epoch parsing (used later for sorting check)
            epoch_keys.append(self._parse_epoch_key_from_line1(l1))

        # All names identical
        first_name = names[0]
        if any(n != first_name for n in names):
            raise ValueError(
                f"Leader TLE series contains multiple names. Expected all names to match '{first_name}', "
                f"but got: {sorted(set(names))}"
            )

        correct_format = True

        # --------------- Check correct sorting (ascending epoch) --------------- #
        for k in range(len(epoch_keys) - 1):
            if epoch_keys[k] >= epoch_keys[k + 1]:
                raise ValueError(
                    f"Leader TLE series is not sorted by ascending epoch. "
                    f"At index {k}: {epoch_keys[k]} > {epoch_keys[k+1]}"
                )

        correct_sorting = True

        
        # Once the leader TLE has been verified, edit flag attribute
        if exists and correct_format and correct_sorting:
            self.verified_leader_TLE_series_file = True
            logging.debug(f"Verified leader TLE series file (Path: {leader_tle_path})")
        else:
            # This should never be reached, but is added to capture any unexpected scenarios
            raise ValueError(f"""The leader TLE series file did not pass verification. Function terminated with flags:
                            exists:          {exists}
                            correct_format:  {correct_format}
                            correct_sorting: {correct_sorting}""")


    def verify_follower_TLE_series_file(self, follower_tle_path: Path) -> bool:
        """
        Verifies the follower TLE series file. 
        If 'True' is returned, the follower TLE file can be used for this simulation run.
        If 'False' is returned, the follower TLE file has to be generated again.

        Checks performed:
            1) File exists and is non-empty
            2) 3-line block structure
            3) Epochs match leader epochs, mean anomaly is shifted correctly and checksum is correct

        Returns:
            True if TLE is verified, false otherwise
        """
        # Set exit flags
        exists = False
        correct_structure_n_data = False

        leader_path = Path(self.leader_tle_series_path)

        # Make sure the leader TLE series file has been verified
        if not self.verified_leader_TLE_series_file:
            logging.debug("Leader TLE series file not verified prior to this method call. Verifying...")
            self.verify_leader_TLE_series_file(leader_path)


        # --------------- 1) Follower TLE exists + is non-empy --------------- #
        if not follower_tle_path.exists():
            logging.debug(f"Follower TLE series file does not exist at (Path: {follower_tle_path}). New follower TLE file(s) must be generated...")
            return False
    
        if follower_tle_path.stat().st_size == 0:
            logging.debug(f"Follower TLE series file exists, but is empty. (Path: {follower_tle_path}). New follower TLE file(s) must be generated...")

        exists = True


        # --------------- 2) 3-line block structure --------------- # 
        try:
            with leader_path.open("r", encoding="utf-8") as f:
                leader_raw = f.readlines()
            with follower_tle_path.open("r", encoding="utf-8") as f:
                follower_raw = f.readlines()
        except Exception as e:
            logging.debug(f"Failed to read leader/follower TLE files: {e}")
            return False
        
        if len(follower_raw) < 3:
            raise ValueError(f"Follower TLE series must contain at least one TLE (3 lines). Got {len(follower_raw)}. New follower TLE file(s) must be generated...")

        if len(follower_raw) % 3 != 0:
            logging.debug(f"Follower TLE series structure invalid (not divisible by 3). New follower TLE file(s) must be generated...")
            return False
        
        if len(follower_raw) != len(leader_raw):
            logging.debug(
                f"Follower TLE series has different length than leader. "
                f"Follower lines={len(follower_raw)}, leader lines={len(leader_raw)}. New follower TLE file(s) must be generated..."
            )
            return False


        # --------------- 3) Epochs match leader epochs, mean anomaly is shifted correctly and checksum is correct --------------- # 
        # Parse follower index from first line
        first_name = self._strip_newline(follower_raw[0]).strip()
        # NOTE: Assuming that generator writes: "follower-{i}"
        if first_name.startswith("follower-"):
            try:
                follower_i = int(first_name.split("-")[1])
            except Exception:
                logging.debug(f"Could not get follower index from first line of (Path: {follower_tle_path}). New follower TLE file(s) must be generated...")
                return False
        else:
            logging.debug(f"Satellite names in follower TLE files does not start with 'follower-'. New follower TLE file(s) must be generated...")
            return False
        
        leader_epoch_keys: list[tuple[int, float]] = []
        follower_epoch_keys: list[tuple[int, float]] = []

        sep_deg = self.inplane_separation_ang
        shift_deg = follower_i * sep_deg

        for b in range(0, len(leader_raw), 3):
            # leader block
            leader_name = self._strip_newline(leader_raw[b]).strip()
            leader_l1 = self._strip_newline(leader_raw[b + 1])
            leader_l2 = self._strip_newline(leader_raw[b + 2])

            # follower block
            follower_name = self._strip_newline(follower_raw[b]).strip()
            follower_l1 = self._strip_newline(follower_raw[b + 1])
            follower_l2 = self._strip_newline(follower_raw[b + 2])

            # Follower name line: require it to be non-empty and consistent across file
            if not follower_name:
                logging.debug(f"Empty follower name at block starting line {b+1} in {follower_tle_path}. New follower TLE file(s) must be generated...")
                return False

            # Basic "standard" checks on follower line1/line2
            if not self._is_tle_line1(follower_l1):
                logging.debug(f"Invalid follower TLE line1 at block starting line {b+1}: {follower_l1!r}. New follower TLE file(s) must be generated...")
                return False
            if not self._is_tle_line2(follower_l2):
                logging.debug(f"Invalid follower TLE line2 at block starting line {b+1}: {follower_l2!r}. New follower TLE file(s) must be generated...")
                return False

            # Checksum validity (both lines)
            try:
                self._validate_checksum(follower_l1, f"Follower TLE line 1 (block starting line {b+1})")
                self._validate_checksum(follower_l2, f"Follower TLE line 2 (block starting line {b+1})")
            except Exception as e:
                logging.debug(f"Follower checksum validation failed: {e}. New follower TLE file(s) must be generated...")
                return False

            # Satnum must match within follower block
            sat1 = self._satnum_from_line(follower_l1)
            sat2 = self._satnum_from_line(follower_l2)
            if not sat1 or not sat2 or sat1 != sat2:
                logging.debug(
                    f"Follower satnum mismatch at block starting line {b+1}: "
                    f"line1 satnum='{sat1}', line2 satnum='{sat2}'. New follower TLE file(s) must be generated..."
                )
                return False

            # Collect epoch keys for leader/follower to compare
            try:
                leader_epoch_keys.append(self._parse_epoch_key_from_line1(leader_l1))
                follower_epoch_keys.append(self._parse_epoch_key_from_line1(follower_l1))
            except Exception as e:
                logging.debug(f"Epoch parsing failed during follower verification: {e}")
                return False
            
            if leader_epoch_keys != follower_epoch_keys:
                logging.debug("Follower epochs do not match leader epochs exactly. New follower TLE file(s) must be generated...")
                return False
            

            # Check for mean anomaly (and checksum) by regenerating expected follower line2
            try:
                M_leader = self.parse_mean_anomaly_from_line2(leader_l2)
                expected_l2 = self.set_mean_anomaly_and_fix_checksum(leader_l2 + "\n", M_leader + shift_deg).rstrip("\n")
            except Exception as e:
                logging.debug(f"Failed generating expected follower line2: {e}")
                return False

            # Compare entire line2 (this implicitly checks mean anomaly AND checksum AND that no other fields changed)
            if follower_l2.rstrip("\n") != expected_l2:
                logging.debug(
                    f"Follower line2 does not match expected shifted version at block starting line {b+1}.\n"
                    f"Expected: {expected_l2!r}\n"
                    f"Got:      {follower_l2!r}. New follower TLE file(s) must be generated..."
                )
                return False

        correct_structure_n_data = True

        # Once the TLE files are verified, log confirmation
        if exists and correct_structure_n_data:
            logging.debug(f"Verified follower TLE series file (Path: {follower_tle_path})")
            return True
        else:
            logging.debug(f"Follower TLE series file (Path: {follower_tle_path}) failed verification. New follower TLE file(s) must be generated...")
            return False
        
    
    def extract_satellite_name_from_TLE(self, tle_path: Path) -> str:
        """
        Returns a satellite name from TLE series file if all names in the file are the same. Raises ValueError otherwise
        """
        try:
            with tle_path.open("r", encoding="utf-8") as f:
                    raw_lines = f.readlines()
        except:
            raise ValueError(f"Failed to open TLE file at (Path: {tle_path})")
        
        # Extract all names
        names: list[str] = []

        for b in range(0, len(raw_lines), 3):
            name = self._strip_newline(raw_lines[b]).strip()

            # Name must exist (non-empty)
            if not name:
                raise ValueError(f"Empty satellite name at block starting line {b+1}")

            names.append(name)

         # All names identical check
        first_name = names[0]
        if any(n != first_name for n in names):
            raise ValueError(
                f"TLE series contains multiple names. Expected all names to match '{first_name}', but got: {sorted(set(names))}"
            )
        
        return first_name


    @staticmethod
    def tle_checksum_char(tle_line: str) -> str:
        """
        Compute the checksum character for a TLE line.
        Standard rule: sum all digits + 1 for each '-' character, mod 10.
        Spaces, dots, plus signs, letters are ignored.
        """
        s = 0
        for ch in tle_line:
            if ch.isdigit():
                s += int(ch)
            elif ch == "-":
                s += 1
        return str(s % 10)


    def set_mean_anomaly_and_fix_checksum(self, line2: str, mean_anom_deg: float) -> str:
        """
        Replace mean anomaly field (cols 44-51, 1-indexed) in TLE line 2
        and fix checksum (last char).
        """
        # Ensure we have a single line without newline
        l2 = line2.rstrip("\n")

        # TLE line2 mean anomaly is in columns 44-51 inclusive (8 chars), 1-indexed.
        # Python 0-index slice: [43:51]
        start, end = 43, 51

        # Wrap to [0, 360)
        mean_anom_wrapped = mean_anom_deg % 360.0

        # Format exactly width 8 with 4 decimals (common for TLE)
        field = f"{mean_anom_wrapped:8.4f}"

        # Replace field
        if len(l2) < 69:
            # If somehow shorter than expected, pad to be safe
            l2 = l2.ljust(69)

        l2_no_csum = l2[:-1]  # everything except existing checksum
        l2_no_csum = l2_no_csum[:start] + field + l2_no_csum[end:]

        # Recompute checksum over the first 68 characters (i.e., excluding checksum char)
        csum = self.tle_checksum_char(l2_no_csum)
        l2_fixed = l2_no_csum + csum

        return l2_fixed + "\n"


    @staticmethod
    def parse_mean_anomaly_from_line2(line2: str) -> float:
        """
        Parse mean anomaly from TLE line 2.
        Safer than fixed-width parsing here: token-based extraction is robust.
        """
        # Example tokens:
        # 2 51053  97.3069  77.1161 0003011 177.8129 182.3139 15.5425... 220845
        toks = line2.strip().split()
        if len(toks) < 8 or toks[0] != "2":
            raise ValueError(f"Unexpected TLE line2 format: {line2!r}")
        return float(toks[6])
    

    @staticmethod
    def _strip_newline(s: str) -> str:
        return s.rstrip("\n")


    @staticmethod
    def _is_tle_line1(l1: str) -> bool:
        return l1.startswith("1 ")


    @staticmethod
    def _is_tle_line2(l2: str) -> bool:
        return l2.startswith("2 ")


    @staticmethod
    def _ensure_min_len(line: str, which: str) -> None:
        # TLE lines are typically 69 chars including checksum at index 68.
        if len(line) < 69:
            raise ValueError(f"{which} is too short (<69 chars): {line!r}")

    
    def _validate_checksum(self, line: str, which: str) -> None:
        # Checksum is last char, computed over first 68 chars.
        self._ensure_min_len(line, which)
        expected = line[68]
        computed = self.tle_checksum_char(line[:68])
        if expected != computed:
            raise ValueError(
                f"{which} checksum mismatch. Expected '{expected}', computed '{computed}'. Line: {line!r}"
            )


    @staticmethod
    def _satnum_from_line(line: str) -> str:
        # Standard sat number field columns 3-7 (1-indexed) => [2:7]
        if len(line) < 7:
            return ""
        return line[2:7].strip()


    def _parse_epoch_key_from_line1(self, l1: str) -> tuple[int, float]:
        """
        Parse epoch from TLE line 1 columns 19-32 (1-indexed),
        format: YYDDD.DDDDDDDD (year, day-of-year with fraction).
        Return a sortable key (full_year, doy_float).
        """
        self._ensure_min_len(l1, "TLE line 1")
        epoch_str = l1[18:32].strip()  # [18:32] corresponds to cols 19-32
        if len(epoch_str) < 5:
            raise ValueError(f"Could not parse epoch from line1: {l1!r}")

        yy = int(epoch_str[0:2])
        doy = float(epoch_str[2:])  # DDD.DDDDDDDD

        # Standard pivot: 57-99 => 1900s, 00-56 => 2000s
        full_year = 1900 + yy if yy >= 57 else 2000 + yy
        return (full_year, doy)