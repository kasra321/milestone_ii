

    def ndc(self, rxcui: str = None) -> pd.DataFrame:
        """
        Get NDC (National Drug Code) data from RxNorm API.
        
        Args:
            rxcui: Optional RxNorm Concept Unique Identifier. 
                  If None, will get NDCs for all medications in the medications table.
        
        Returns:
            pd.DataFrame: DataFrame containing rxcui and associated NDC codes
        """
        base_url = "https://rxnav.nlm.nih.gov/REST"
        
        if rxcui:
            # Get NDCs for a specific rxcui
            ndc_url = f"{base_url}/rxcui/{rxcui}/ndcs.json"
            response = requests.get(ndc_url)
            
            if not response.ok:
                raise ConnectionError(f"Failed to fetch NDCs: {response.status_code}")
                
            ndc_data = response.json()
            ndcs = ndc_data.get("ndcGroup", {}).get("ndcList", {}).get("ndc", [])
            
            return pd.DataFrame({
                "rxcui": [rxcui] * len(ndcs),
                "ndc": ndcs
            })
        else:
            # Get medications from the medications table
            meds = self.medications()
            unique_names = meds['name'].unique()
            
            all_ndcs = []
            for name in unique_names:
                # First get rxcui for the medication name
                search_url = f"{base_url}/rxcui.json"
                params = {"name": name}
                response = requests.get(search_url, params=params)
                
                if not response.ok:
                    print(f"Failed to get rxcui for {name}: {response.status_code}")
                    continue
                    
                data = response.json()
                rxcui_list = data.get("idGroup", {}).get("rxnormId", [])
                
                if not rxcui_list:
                    print(f"No rxcui found for {name}")
                    continue
                
                # Get NDCs for each rxcui
                for rxcui in rxcui_list:
                    try:
                        ndc_df = self.ndc(rxcui)
                        ndc_df['medication_name'] = name
                        all_ndcs.append(ndc_df)
                        # Add a small delay to avoid overwhelming the API
                        sleep(0.1)
                    except Exception as e:
                        print(f"Failed to fetch NDCs for {name} (rxcui: {rxcui}): {str(e)}")
                        continue
            
            if all_ndcs:
                return pd.concat(all_ndcs, ignore_index=True)
            else:
                return pd.DataFrame(columns=['rxcui', 'ndc', 'medication_name'])
    
    def get_all_ndcs(self) -> pd.DataFrame:
    
        """
        Get all available NDC codes from RxNorm API.
        
        Returns:
            pd.DataFrame: DataFrame containing rxcui and NDC codes
        """
    
        base_url = "https://rxnav.nlm.nih.gov/REST"
        
        # First get all RxCUIs
        all_rxcui_url = f"{base_url}/allconcepts.json?tty=SCD+SBD"  # Get only clinical drugs
        response = requests.get(all_rxcui_url)
        
        if not response.ok:
            raise ConnectionError(f"Failed to fetch RxCUIs: {response.status_code}")
        
        data = response.json()
        concepts = data.get("minConceptGroup", {}).get("minConcept", [])
        
        all_ndcs = []
        
        # For each RxCUI, get its NDCs
        for concept in concepts:
            rxcui = concept.get("rxcui")
            name = concept.get("name")
            
            # Get NDCs for this RxCUI
            ndc_url = f"{base_url}/rxcui/{rxcui}/ndcs.json"
            response = requests.get(ndc_url)
            
            if not response.ok:
                print(f"Failed to fetch NDCs for {rxcui}: {response.status_code}")
                continue
                
            ndc_data = response.json()
            ndcs = ndc_data.get("ndcGroup", {}).get("ndcList", {}).get("ndc", [])
            
            if ndcs:
                df = pd.DataFrame({
                    "rxcui": [rxcui] * len(ndcs),
                    "medication_name": [name] * len(ndcs),
                    "ndc": ndcs
                })
                all_ndcs.append(df)
            
            # Add a small delay to avoid overwhelming the API
            sleep(0.1)
        
        if all_ndcs:
            return pd.concat(all_ndcs, ignore_index=True)
        else:
            return pd.DataFrame(columns=['rxcui', 'medication_name', 'ndc'])