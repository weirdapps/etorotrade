import yfinance as yf
import pandas as pd
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_sp500_constituents():
    """Get S&P 500 constituents from Wikipedia (most reliable source)."""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        df = pd.read_html(url)[0]
        return df['Symbol'].tolist()
    except Exception as e:
        logger.error(f"Error fetching S&P 500 constituents: {str(e)}")
        return []

def get_ftse100_constituents():
    """Get FTSE 100 constituents."""
    try:
        symbols = [
            'AAL.L', 'ABDN.L', 'ABF.L', 'ADM.L', 'AHT.L', 'ANTO.L', 'AUTO.L', 'AV.L', 'AZN.L', 'BA.L',
            'BARC.L', 'BATS.L', 'BDEV.L', 'BEZ.L', 'BKG.L', 'BLND.L', 'BME.L', 'BNZL.L', 'BP.L', 'BRBY.L',
            'BT-A.L', 'CCH.L', 'CNA.L', 'CPG.L', 'CRDA.L', 'DCC.L', 'DGE.L', 'GLEN.L', 'GSK.L', 'HIK.L',
            'HLMA.L', 'HSBA.L', 'IAG.L', 'IHG.L', 'IMB.L', 'INF.L', 'ITRK.L', 'JD.L', 'KGF.L', 'LAND.L',
            'LGEN.L', 'LLOY.L', 'LMT.L', 'MNG.L', 'MRO.L', 'NG.L', 'NWG.L', 'NXT.L', 'OCDO.L', 'PRU.L',
            'PSN.L', 'PSON.L', 'RIO.L', 'RKT.L', 'RMV.L', 'RR.L', 'RS1.L', 'SBRY.L', 'SDR.L', 'SGE.L',
            'SGRO.L', 'SHEL.L', 'SKG.L', 'SMDS.L', 'SMIN.L', 'SMT.L', 'SN.L', 'SPX.L', 'SSE.L', 'STAN.L',
            'STJ.L', 'SVT.L', 'TSCO.L', 'TW.L', 'ULVR.L', 'UU.L', 'VOD.L', 'WEIR.L', 'WPP.L', 'WTB.L'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with FTSE 100 constituents: {str(e)}")
        return []

def get_cac40_constituents():
    """Get CAC 40 constituents."""
    try:
        symbols = [
            'AC.PA', 'AI.PA', 'AIR.PA', 'ALO.PA', 'ATO.PA', 'BN.PA', 'BNP.PA', 'CA.PA', 'CAP.PA', 'CS.PA',
            'DG.PA', 'DSY.PA', 'EN.PA', 'ENGI.PA', 'ERF.PA', 'EL.PA', 'FTI.PA', 'GLE.PA', 'HO.PA', 'KER.PA',
            'LR.PA', 'MC.PA', 'ML.PA', 'OR.PA', 'ORA.PA', 'PUB.PA', 'RI.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA',
            'SAN.PA', 'SU.PA', 'STM.PA', 'TEP.PA', 'TTE.PA', 'URW.PA', 'VIE.PA', 'VIV.PA', 'WLN.PA'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with CAC 40 constituents: {str(e)}")
        return []

def get_dax_constituents():
    """Get DAX constituents."""
    try:
        symbols = [
            'ADS.DE', 'AIR.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BMW.DE', 'BNR.DE', 'CON.DE', 'DTG.DE', 'DB1.DE',
            'DBK.DE', 'DHL.DE', 'DTE.DE', 'EOAN.DE', 'FRE.DE', 'HEI.DE', 'HEN3.DE', 'IFX.DE', 'LIN.DE', 'MBG.DE',
            'MRK.DE', 'MTX.DE', 'MUV2.DE', 'PAH3.DE', 'PUM.DE', 'RWE.DE', 'SAP.DE', 'SHL.DE', 'SIE.DE', 'SRT3.DE',
            'VNA.DE', 'VOW3.DE'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with DAX constituents: {str(e)}")
        return []

def get_nikkei225_constituents():
    """Get Nikkei 225 constituents."""
    try:
        symbols = [
            '1332.T', '1605.T', '1721.T', '1801.T', '1802.T', '1803.T', '1808.T', '1812.T', '1925.T', '1928.T',
            '1963.T', '2002.T', '2269.T', '2282.T', '2413.T', '2432.T', '2501.T', '2502.T', '2503.T', '2531.T',
            '2768.T', '2801.T', '2802.T', '2871.T', '2914.T', '3086.T', '3105.T', '3401.T', '3402.T', '3405.T',
            '3407.T', '3436.T', '3861.T', '3863.T', '4004.T', '4005.T', '4021.T', '4042.T', '4043.T', '4061.T',
            '4151.T', '4183.T', '4188.T', '4208.T', '4324.T', '4452.T', '4502.T', '4503.T', '4506.T', '4507.T',
            '4519.T', '4523.T', '4543.T', '4568.T', '4578.T', '4631.T', '4689.T', '4704.T', '4751.T', '4755.T',
            '4901.T', '4902.T', '4911.T', '5019.T', '5020.T', '5101.T', '5108.T', '5201.T', '5202.T', '5214.T',
            '5232.T', '5233.T', '5301.T', '5332.T', '5333.T', '5401.T', '5406.T', '5411.T', '5541.T', '5631.T',
            '5706.T', '5707.T', '5711.T', '5713.T', '5714.T', '5801.T', '5802.T', '5803.T', '5831.T', '6098.T',
            '6103.T', '6113.T', '6178.T', '6301.T', '6302.T', '6305.T', '6326.T', '6361.T', '6367.T', '6471.T',
            '6472.T', '6473.T', '6479.T', '6501.T', '6503.T', '6504.T', '6506.T', '6645.T', '6674.T', '6701.T',
            '6702.T', '6703.T', '6724.T', '6752.T', '6753.T', '6758.T', '6762.T', '6770.T', '6841.T', '6857.T',
            '6902.T', '6952.T', '6954.T', '6971.T', '6976.T', '6988.T', '7003.T', '7004.T', '7011.T', '7012.T',
            '7013.T', '7186.T', '7201.T', '7202.T', '7203.T', '7205.T', '7211.T', '7261.T', '7267.T', '7269.T',
            '7270.T', '7272.T', '7731.T', '7733.T', '7735.T', '7751.T', '7752.T', '7762.T', '7832.T', '7911.T',
            '7912.T', '7951.T', '8001.T', '8002.T', '8015.T', '8031.T', '8035.T', '8053.T', '8058.T', '8233.T',
            '8252.T', '8253.T', '8267.T', '8303.T', '8304.T', '8306.T', '8308.T', '8309.T', '8316.T', '8331.T',
            '8354.T', '8355.T', '8411.T', '8601.T', '8604.T', '8628.T', '8630.T', '8725.T', '8729.T', '8750.T',
            '8766.T', '8795.T', '8801.T', '8802.T', '8804.T', '8830.T', '9001.T', '9005.T', '9007.T', '9008.T',
            '9009.T', '9020.T', '9021.T', '9022.T', '9062.T', '9064.T', '9101.T', '9104.T', '9107.T', '9202.T',
            '9301.T', '9432.T', '9433.T', '9434.T', '9501.T', '9502.T', '9503.T', '9531.T', '9532.T', '9602.T',
            '9613.T', '9735.T', '9766.T', '9983.T', '9984.T'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with Nikkei 225 constituents: {str(e)}")
        return []

def get_hangseng_constituents():
    """Get Hang Seng constituents."""
    try:
        symbols = [
            '0001.HK', '0002.HK', '0003.HK', '0005.HK', '0006.HK', '0011.HK', '0012.HK', '0016.HK', '0017.HK', '0019.HK',
            '0027.HK', '0066.HK', '0101.HK', '0175.HK', '0241.HK', '0267.HK', '0288.HK', '0291.HK', '0386.HK', '0388.HK',
            '0669.HK', '0688.HK', '0700.HK', '0762.HK', '0823.HK', '0857.HK', '0883.HK', '0939.HK', '0941.HK', '0960.HK',
            '0968.HK', '0981.HK', '0992.HK', '1038.HK', '1044.HK', '1093.HK', '1109.HK', '1113.HK', '1177.HK', '1299.HK',
            '1398.HK', '1810.HK', '1876.HK', '1928.HK', '1997.HK', '2007.HK', '2018.HK', '2020.HK', '2269.HK', '2313.HK',
            '2318.HK', '2319.HK', '2331.HK', '2382.HK', '2388.HK', '2628.HK', '3690.HK', '3968.HK', '3988.HK', '6098.HK',
            '6862.HK', '9618.HK', '9633.HK', '9888.HK', '9961.HK', '9988.HK', '9999.HK'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with Hang Seng constituents: {str(e)}")
        return []

def save_constituents_to_csv(all_symbols):
    """
    Save constituent symbols to a CSV file.
    
    Args:
        all_symbols (list): List of all constituent symbols
    """
    try:
        # Create input directory if it doesn't exist
        input_dir = Path(__file__).parent / 'input'
        input_dir.mkdir(exist_ok=True)
        
        # Remove duplicates and sort
        unique_symbols = sorted(set(all_symbols))
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame({'Symbol': unique_symbols})
        filepath = input_dir / 'cons.csv'
        df.to_csv(filepath, index=False)
        logger.info(f"Successfully saved {len(unique_symbols)} unique constituents to {filepath}")
    except Exception as e:
        logger.error(f"Error saving constituents: {str(e)}")

def main():
    """
    Main function to fetch and save constituents for major indices.
    """
    all_symbols = []
    
    # US Indices
    logger.info("Fetching S&P 500 constituents")
    all_symbols.extend(get_sp500_constituents())
    
    # European Indices
    logger.info("Fetching FTSE 100 constituents")
    all_symbols.extend(get_ftse100_constituents())
    
    logger.info("Fetching CAC 40 constituents")
    all_symbols.extend(get_cac40_constituents())
    
    logger.info("Fetching DAX constituents")
    all_symbols.extend(get_dax_constituents())
    
    # Asian Indices
    logger.info("Fetching Nikkei 225 constituents")
    all_symbols.extend(get_nikkei225_constituents())
    
    logger.info("Fetching Hang Seng constituents")
    all_symbols.extend(get_hangseng_constituents())
    
    save_constituents_to_csv(all_symbols)

if __name__ == "__main__":
    main()