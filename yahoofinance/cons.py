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

def get_ibex_constituents():
    """Get IBEX 35 constituents (Madrid Stock Exchange)."""
    try:
        symbols = [
            'ACS.MC', 'ACX.MC', 'AENA.MC', 'AMS.MC', 'ANA.MC', 'BBVA.MC', 'BKT.MC', 'CABK.MC', 'CLNX.MC', 'COL.MC',
            'ELE.MC', 'ENG.MC', 'FDR.MC', 'FER.MC', 'GRF.MC', 'IAG.MC', 'IBE.MC', 'IDR.MC', 'ITX.MC', 'LOG.MC',
            'MAP.MC', 'MEL.MC', 'MRL.MC', 'NTGY.MC', 'PHM.MC', 'RED.MC', 'REP.MC', 'ROVI.MC', 'SAB.MC', 'SAN.MC',
            'SCYR.MC', 'SLR.MC', 'TEF.MC', 'UNI.MC', 'VIS.MC'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with IBEX constituents: {str(e)}")
        return []

def get_ftsemib_constituents():
    """Get FTSE MIB constituents (Italian Stock Exchange)."""
    try:
        symbols = [
            'A2A.MI', 'AMP.MI', 'ATL.MI', 'AZM.MI', 'BAMI.MI', 'BMED.MI', 'BMPS.MI', 'BPE.MI', 'BZU.MI', 'CPR.MI',
            'DIA.MI', 'ENEL.MI', 'ENI.MI', 'EXO.MI', 'FBK.MI', 'FCA.MI', 'G.MI', 'HER.MI', 'IG.MI', 'INW.MI',
            'ISP.MI', 'IVG.MI', 'LDO.MI', 'MB.MI', 'MONC.MI', 'NEXI.MI', 'PIRC.MI', 'PRY.MI', 'PST.MI', 'RACE.MI',
            'REC.MI', 'SPM.MI', 'SRG.MI', 'STM.MI', 'TEN.MI', 'TIT.MI', 'TRN.MI', 'UCG.MI', 'UNI.MI', 'WBD.MI'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with FTSE MIB constituents: {str(e)}")
        return []

def get_psi_constituents():
    """Get PSI 20 constituents (Lisbon Stock Exchange)."""
    try:
        symbols = [
            'ALTR.LS', 'BCP.LS', 'COR.LS', 'CTT.LS', 'EDP.LS', 'EDPR.LS', 'ESON.LS', 'EGL.LS', 'GALP.LS', 'GRE.LS',
            'IBS.LS', 'JMT.LS', 'NBA.LS', 'NOS.LS', 'NVG.LS', 'RAM.LS', 'RENE.LS', 'SEM.LS', 'SON.LS'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with PSI constituents: {str(e)}")
        return []

def get_smi_constituents():
    """Get Swiss Market Index (SMI) constituents."""
    try:
        symbols = [
            'ABBN.SW', 'ADEN.SW', 'ALV.SW', 'CFR.SW', 'CSGN.SW', 'GEBN.SW', 'GIVN.SW', 'HOLN.SW', 'LONN.SW', 'NESN.SW',
            'NOVN.SW', 'ROG.SW', 'SCMN.SW', 'SGSN.SW', 'SIKA.SW', 'SLHN.SW', 'SOON.SW', 'SREN.SW', 'UHR.SW', 'UBSG.SW'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with SMI constituents: {str(e)}")
        return []

def get_omxc25_constituents():
    """Get OMXC25 constituents (Copenhagen Stock Exchange)."""
    try:
        symbols = [
            'AMBU-B.CO', 'CARL-B.CO', 'CHR.CO', 'COLO-B.CO', 'DANSKE.CO', 'DEMANT.CO', 'DSV.CO', 'FLS.CO', 'GEN.CO',
            'GN.CO', 'ISS.CO', 'JYSK.CO', 'MAERSK-A.CO', 'MAERSK-B.CO', 'NOVO-B.CO', 'NZYM-B.CO', 'ORSTED.CO',
            'PNDORA.CO', 'RBREW.CO', 'ROCK-B.CO', 'SIM.CO', 'TOP.CO', 'TRYG.CO', 'VWS.CO', 'WDH.CO'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with OMXC25 constituents: {str(e)}")
        return []

def get_athex_constituents():
    """Get ATHEX constituents (Athens Stock Exchange)."""
    try:
        symbols = [
            'ALPHA.AT', 'ADMIE.AT', 'AEGN.AT', 'OPAP.AT', 'PPC.AT', 'ETE.AT', 'EUROB.AT', 'GEKTERNA.AT', 'HTO.AT',
            'LAMDA.AT', 'MOH.AT', 'MYTIL.AT', 'OTE.AT', 'PEIR.AT', 'TPEIR.AT', 'TENERGY.AT', 'TITC.AT', 'VIOHA.AT'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with ATHEX constituents: {str(e)}")
        return []

def get_nasdaq100_constituents():
    """Get NASDAQ-100 constituents from Wikipedia."""
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        dfs = pd.read_html(url)
        for df in dfs:
            if 'Ticker' in df.columns:
                symbols = df['Ticker'].tolist()
                logger.info(f"Found {len(symbols)} NASDAQ-100 constituents")
                return symbols
        
        # Fallback if structure changes
        logger.warning("Table structure on Wikipedia for NASDAQ-100 may have changed")
        url = 'https://www.nasdaq.com/market-activity/quotes/nasdaq-100-index-components'
        df = pd.read_html(url)[0]
        symbols = df['Symbol'].tolist()
        return symbols
    except Exception as e:
        logger.error(f"Error fetching NASDAQ-100 constituents: {str(e)}")
        return []

def get_dowjones_constituents():
    """Get Dow Jones Industrial Average (DJIA) constituents from Wikipedia."""
    try:
        url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
        dfs = pd.read_html(url)
        for df in dfs:
            if 'Symbol' in df.columns:
                symbols = df['Symbol'].tolist()
                logger.info(f"Found {len(symbols)} Dow Jones constituents")
                return symbols
        
        # Fallback
        logger.warning("Table structure on Wikipedia for Dow Jones may have changed")
        return [
            'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
            'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
            'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
        ]
    except Exception as e:
        logger.error(f"Error fetching Dow Jones constituents: {str(e)}")
        return []

def get_russell2000_constituents():
    """Get Russell 2000 constituents from external sources.
    
    Since the Russell 2000 has 2000 stocks and is frequently rebalanced,
    we fetch it from reliable financial data sources."""
    try:
        # Try to fetch from ETF holdings (IWM is the iShares Russell 2000 ETF)
        try:
            url = 'https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax'
            response = pd.read_json(url)
            if 'holdings' in response.columns:
                holdings_data = response['holdings'].iloc[0]
                if isinstance(holdings_data, list) and len(holdings_data) > 0:
                    symbols = [holding['ticker'] for holding in holdings_data if 'ticker' in holding]
                    logger.info(f"Found {len(symbols)} Russell 2000 constituents from iShares")
                    return symbols
        except Exception as inner_e:
            logger.warning(f"Could not fetch Russell 2000 from primary source: {str(inner_e)}")

        # Alternative method - fetch from financial data provider
        try:
            url = 'https://stockmarketmba.com/stocksintheindexrussell2000.php'
            tables = pd.read_html(url)
            for table in tables:
                if 'Symbol' in table.columns:
                    symbols = table['Symbol'].tolist()
                    logger.info(f"Found {len(symbols)} Russell 2000 constituents from alternative source")
                    return symbols
        except Exception as inner_e:
            logger.warning(f"Could not fetch Russell 2000 from secondary source: {str(inner_e)}")
            
        # Fallback to a smaller representation of the Russell 2000 index
        logger.warning("Using fallback data for Russell 2000 (partial list)")
        # Use iShares Russell 2000 ETF (IWM) top holdings as representative
        symbols = [
            'AAON', 'ABCB', 'ACAD', 'ACHC', 'ACIW', 'ACLS', 'ACM', 'ADTN', 'AEIS', 'AEVA',
            'AGO', 'AINV', 'AJRD', 'AKRO', 'ALEC', 'ALGT', 'ALKS', 'ALLO', 'ALPN', 'ALTG',
            'AMC', 'AMKR', 'ANIK', 'AOSL', 'APPF', 'APPH', 'ARCB', 'ARCT', 'ARDX', 'ARNA',
            # Add more symbols here from top holdings
        ]
        
        # Alternatively, try to download from a Russell 2000 ETF
        try:
            iwm = yf.Ticker("IWM")
            holdings = iwm.get_holdings()
            if len(holdings) > 0 and 'ticker' in holdings.columns:
                symbols = holdings['ticker'].tolist()
                logger.info(f"Found {len(symbols)} Russell 2000 constituents from Yahoo Finance ETF holdings")
                return symbols
        except Exception as inner_e:
            logger.warning(f"Could not fetch Russell 2000 from YF ETF holdings: {str(inner_e)}")
        
        return symbols
    except Exception as e:
        logger.error(f"Error with Russell 2000 constituents: {str(e)}")
        return []

def get_csi300_constituents():
    """Get CSI 300 constituents (China)."""
    try:
        # Try to fetch from international financial data sources
        try:
            # First try to get from a financial data source
            url = "https://en.wikipedia.org/wiki/CSI_300_Index"
            dfs = pd.read_html(url)
            for df in dfs:
                if 'Symbol' in df.columns:
                    # Extract tickers and append appropriate exchange suffix
                    symbols = []
                    for _, row in df.iterrows():
                        # Check exchange column if available, otherwise use a heuristic
                        if 'Symbol' in df.columns and 'Exchange' in df.columns:
                            symbol = row['Symbol']
                            exchange = row['Exchange']
                            if exchange == 'Shanghai':
                                symbols.append(f"{symbol}.SS")
                            elif exchange == 'Shenzhen':
                                symbols.append(f"{symbol}.SZ")
                        elif 'Ticker' in df.columns:
                            symbol = row['Ticker']
                            # Apply heuristics for exchange suffix
                            if symbol.startswith('6') or symbol.startswith('9'):
                                symbols.append(f"{symbol}.SS")  # Shanghai
                            elif symbol.startswith('0') or symbol.startswith('3'):
                                symbols.append(f"{symbol}.SZ")  # Shenzhen
                    
                    logger.info(f"Found {len(symbols)} CSI 300 constituents")
                    return symbols
        except Exception as inner_e:
            logger.warning(f"Could not fetch CSI 300 from primary source: {str(inner_e)}")
        
        # Try alternative source
        try:
            url = "https://www.investing.com/indices/csi300-components"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = pd.read_html(url, headers=headers)
            if len(response) > 0 and 'Symbol' in response[0].columns:
                symbols = response[0]['Symbol'].tolist()
                logger.info(f"Found {len(symbols)} CSI 300 constituents from alternative source")
                return symbols
        except Exception as inner_e:
            logger.warning(f"Could not fetch CSI 300 from secondary source: {str(inner_e)}")
                    
        # If web scraping fails, use a comprehensive but possibly not 100% current list
        logger.warning("Using fallback data for CSI 300 (comprehensive but may not be the latest composition)")
        symbols = [
            # Shanghai Stock Exchange (SSE)
            '600000.SS', '600009.SS', '600016.SS', '600018.SS', '600019.SS', '600025.SS', '600027.SS',
            '600028.SS', '600029.SS', '600030.SS', '600031.SS', '600036.SS', '600038.SS', '600048.SS',
            '600050.SS', '600061.SS', '600066.SS', '600085.SS', '600104.SS', '600109.SS', '600111.SS',
            '600115.SS', '600150.SS', '600161.SS', '600176.SS', '600183.SS', '600196.SS', '600208.SS',
            '600219.SS', '600233.SS', '600271.SS', '600276.SS', '600297.SS', '600299.SS', '600309.SS',
            '600332.SS', '600340.SS', '600346.SS', '600352.SS', '600362.SS', '600369.SS', '600372.SS',
            '600383.SS', '600390.SS', '600398.SS', '600406.SS', '600436.SS', '600438.SS', '600482.SS',
            '600487.SS', '600489.SS', '600498.SS', '600516.SS', '600519.SS', '600547.SS', '600570.SS',
            '600584.SS', '600585.SS', '600588.SS', '600606.SS', '600655.SS', '600660.SS', '600674.SS',
            '600690.SS', '600703.SS', '600705.SS', '600741.SS', '600760.SS', '600795.SS', '600809.SS',
            '600837.SS', '600886.SS', '600887.SS', '600893.SS', '600900.SS', '600919.SS', '600926.SS',
            '600958.SS', '600989.SS', '600999.SS', '601006.SS', '601009.SS', '601012.SS', '601018.SS',
            '601021.SS', '601066.SS', '601088.SS', '601108.SS', '601111.SS', '601138.SS', '601155.SS',
            '601162.SS', '601166.SS', '601169.SS', '601186.SS', '601198.SS', '601211.SS', '601216.SS',
            '601225.SS', '601229.SS', '601236.SS', '601238.SS', '601288.SS', '601318.SS', '601319.SS',
            '601328.SS', '601336.SS', '601360.SS', '601377.SS', '601390.SS', '601398.SS', '601555.SS',
            '601577.SS', '601600.SS', '601601.SS', '601607.SS', '601618.SS', '601628.SS', '601633.SS',
            '601658.SS', '601668.SS', '601669.SS', '601688.SS', '601696.SS', '601698.SS', '601727.SS',
            '601766.SS', '601788.SS', '601800.SS', '601808.SS', '601816.SS', '601818.SS', '601828.SS',
            '601838.SS', '601857.SS', '601877.SS', '601878.SS', '601881.SS', '601888.SS', '601898.SS',
            '601899.SS', '601901.SS', '601916.SS', '601919.SS', '601933.SS', '601939.SS', '601985.SS',
            '601988.SS', '601989.SS', '601995.SS', '601998.SS', '603019.SS', '603160.SS', '603259.SS',
            '603260.SS', '603288.SS', '603369.SS', '603501.SS', '603658.SS', '603833.SS', '603899.SS',
            '603986.SS', '603993.SS', '688005.SS', '688008.SS', '688012.SS', '688036.SS', '688111.SS',
            '688363.SS', '688396.SS',
            
            # Shenzhen Stock Exchange (SZSE)
            '000001.SZ', '000002.SZ', '000063.SZ', '000066.SZ', '000069.SZ', '000100.SZ', '000157.SZ',
            '000166.SZ', '000333.SZ', '000338.SZ', '000425.SZ', '000538.SZ', '000568.SZ', '000596.SZ',
            '000625.SZ', '000627.SZ', '000651.SZ', '000661.SZ', '000703.SZ', '000708.SZ', '000723.SZ',
            '000725.SZ', '000768.SZ', '000776.SZ', '000783.SZ', '000786.SZ', '000800.SZ', '000858.SZ',
            '000876.SZ', '000895.SZ', '000938.SZ', '000963.SZ', '000977.SZ', '001979.SZ', '002001.SZ',
            '002007.SZ', '002008.SZ', '002024.SZ', '002027.SZ', '002044.SZ', '002050.SZ', '002120.SZ',
            '002129.SZ', '002142.SZ', '002179.SZ', '002202.SZ', '002230.SZ', '002236.SZ', '002241.SZ',
            '002252.SZ', '002271.SZ', '002304.SZ', '002311.SZ', '002352.SZ', '002371.SZ', '002410.SZ',
            '002415.SZ', '002459.SZ', '002460.SZ', '002475.SZ', '002493.SZ', '002555.SZ', '002594.SZ',
            '002601.SZ', '002602.SZ', '002607.SZ', '002624.SZ', '002673.SZ', '002714.SZ', '002736.SZ',
            '002812.SZ', '002821.SZ', '002938.SZ', '002945.SZ', '002958.SZ', '003816.SZ', '300003.SZ',
            '300014.SZ', '300015.SZ', '300033.SZ', '300059.SZ', '300122.SZ', '300124.SZ', '300136.SZ',
            '300142.SZ', '300207.SZ', '300223.SZ', '300274.SZ', '300316.SZ', '300347.SZ', '300408.SZ',
            '300413.SZ', '300433.SZ', '300450.SZ', '300498.SZ', '300529.SZ', '300558.SZ', '300595.SZ',
            '300616.SZ', '300628.SZ', '300661.SZ', '300750.SZ', '300760.SZ', '300866.SZ', '300896.SZ',
            '300919.SZ', '300999.SZ'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with CSI 300 constituents: {str(e)}")
        return []

def get_kospi_constituents():
    """Get KOSPI constituents (South Korea)."""
    try:
        # Try to fetch from Wikipedia or other reliable sources
        try:
            url = "https://en.wikipedia.org/wiki/KOSPI"
            dfs = pd.read_html(url)
            for df in dfs:
                if 'Symbol' in df.columns:
                    # Format tickers with .KS suffix
                    symbols = [f"{symbol}.KS" for symbol in df['Symbol'].tolist()]
                    logger.info(f"Found {len(symbols)} KOSPI constituents")
                    return symbols
        except Exception as inner_e:
            logger.warning(f"Could not fetch KOSPI from primary source: {str(inner_e)}")
        
        # Try alternative source - financial data provider
        try:
            url = "https://www.investing.com/indices/kospi-components"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = pd.read_html(url, headers=headers)
            if len(response) > 0 and 'Symbol' in response[0].columns:
                symbols = [f"{symbol}.KS" for symbol in response[0]['Symbol'].tolist()]
                logger.info(f"Found {len(symbols)} KOSPI constituents from alternative source")
                return symbols
        except Exception as inner_e:
            logger.warning(f"Could not fetch KOSPI from secondary source: {str(inner_e)}")
            
        # Try from ETF
        try:
            # Check for KOSPI ETF (EWY is iShares MSCI South Korea ETF)
            ewy = yf.Ticker("EWY")
            holdings = ewy.get_holdings()
            if len(holdings) > 0 and 'ticker' in holdings.columns:
                # Filter for Korean stocks (.KS suffix)
                symbols = [ticker for ticker in holdings['ticker'].tolist() if '.KS' in ticker]
                if len(symbols) > 0:
                    logger.info(f"Found {len(symbols)} KOSPI constituents from ETF holdings")
                    return symbols
        except Exception as inner_e:
            logger.warning(f"Could not fetch KOSPI from ETF holdings: {str(inner_e)}")
        
        # Comprehensive fallback list with all major KOSPI stocks
        logger.warning("Using fallback data for KOSPI (comprehensive but may not be the latest composition)")
        symbols = [
            '005930.KS', '000660.KS', '005380.KS', '005490.KS', '051910.KS', '006400.KS', '035420.KS', '028260.KS', '000270.KS', '068270.KS',
            '005935.KS', '207940.KS', '012330.KS', '055550.KS', '066570.KS', '015760.KS', '034730.KS', '096770.KS', '032830.KS', '003550.KS',
            '017670.KS', '018260.KS', '009150.KS', '086790.KS', '011170.KS', '036570.KS', '033780.KS', '010130.KS', '316140.KS', '009540.KS',
            '000810.KS', '024110.KS', '010950.KS', '051900.KS', '003670.KS', '035720.KS', '012750.KS', '105560.KS', '000720.KS', '161390.KS',
            '006800.KS', '096530.KS', '323410.KS', '008770.KS', '051900.KS', '003490.KS', '032640.KS', '069500.KS', '251270.KS', '078930.KS',
            '006280.KS', '000990.KS', '034220.KS', '011070.KS', '003670.KS', '064350.KS', '001040.KS', '030200.KS', '009540.KS', '267250.KS',
            '011780.KS', '003410.KS', '000100.KS', '005850.KS', '006650.KS', '047810.KS', '018880.KS', '097950.KS', '079550.KS', '081660.KS',
            '005440.KS', '139480.KS', '138040.KS', '010950.KS', '007070.KS', '001450.KS', '004020.KS', '009240.KS', '111770.KS', '029780.KS',
            '001740.KS', '071050.KS', '004170.KS', '010140.KS', '016360.KS', '000720.KS', '003230.KS', '002380.KS', '214320.KS', '011170.KS',
            '114090.KS', '009150.KS', '000210.KS', '012750.KS', '002790.KS', '039490.KS', '003850.KS', '036460.KS', '006360.KS', '023530.KS'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with KOSPI constituents: {str(e)}")
        return []

def get_nifty50_constituents():
    """Get NIFTY 50 constituents (India)."""
    try:
        symbols = [
            'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'HDFC.NS', 'TCS.NS', 'ITC.NS', 'KOTAKBANK.NS', 'LT.NS', 'HINDUNILVR.NS',
            'SBIN.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'HCLTECH.NS', 'ULTRACEMCO.NS',
            'BAJAJFINSV.NS', 'ADANIPORTS.NS', 'NESTLEIND.NS', 'TATAMOTORS.NS', 'NTPC.NS', 'POWERGRID.NS', 'WIPRO.NS', 'TECHM.NS', 'INDUSINDBK.NS', 'M&M.NS',
            'TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'ADANIENT.NS', 'ONGC.NS', 'CIPLA.NS', 'DRREDDY.NS', 'GRASIM.NS', 'COALINDIA.NS', 'EICHERMOT.NS',
            'HDFCLIFE.NS', 'DIVISLAB.NS', 'SBILIFE.NS', 'UPL.NS', 'TATACONSUM.NS', 'BRITANNIA.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS', 'IOC.NS', 'BPCL.NS'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with NIFTY 50 constituents: {str(e)}")
        return []

def get_asx200_constituents():
    """Get ASX 200 constituents (Australia).
    Note: This returns a subset of prominent ASX 200 constituents."""
    try:
        symbols = [
            'BHP.AX', 'CBA.AX', 'CSL.AX', 'NAB.AX', 'WBC.AX', 'ANZ.AX', 'FMG.AX', 'MQG.AX', 'WES.AX', 'TLS.AX',
            'RIO.AX', 'WOW.AX', 'NCM.AX', 'TCL.AX', 'GMG.AX', 'WPL.AX', 'STO.AX', 'SUN.AX', 'REA.AX', 'COL.AX',
            'ASX.AX', 'SCG.AX', 'AMC.AX', 'QBE.AX', 'ALL.AX', 'TWE.AX', 'CPU.AX', 'IAG.AX', 'JHX.AX', 'ORG.AX',
            'AMP.AX', 'APA.AX', 'BOQ.AX', 'BXB.AX', 'CCL.AX', 'CGF.AX', 'CIM.AX', 'COH.AX', 'DXS.AX', 'FLT.AX',
            'FPH.AX', 'GPT.AX', 'HVN.AX', 'IPL.AX', 'JBH.AX', 'LLC.AX', 'MGR.AX', 'MPL.AX', 'NST.AX', 'OSH.AX'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with ASX 200 constituents: {str(e)}")
        return []

def get_taiex_constituents():
    """Get TAIEX constituents (Taiwan).
    Note: This returns a subset of prominent TAIEX constituents."""
    try:
        symbols = [
            '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2303.TW', '2882.TW', '1301.TW', '2881.TW', '1303.TW', '2002.TW',
            '2886.TW', '2891.TW', '3711.TW', '2382.TW', '2412.TW', '3045.TW', '2201.TW', '1216.TW', '2207.TW', '2880.TW',
            '2885.TW', '1326.TW', '2884.TW', '2301.TW', '2409.TW', '2357.TW', '3008.TW', '2379.TW', '2892.TW', '1101.TW',
            '2474.TW', '2311.TW', '2395.TW', '4938.TW', '2887.TW', '2890.TW', '6505.TW', '3034.TW', '2106.TW', '2603.TW'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with TAIEX constituents: {str(e)}")
        return []

def get_tsx_constituents():
    """Get TSX Composite constituents (Canada).
    Note: This returns a subset of prominent TSX Composite constituents."""
    try:
        symbols = [
            'RY.TO', 'TD.TO', 'ENB.TO', 'CNR.TO', 'CP.TO', 'BNS.TO', 'BMO.TO', 'CM.TO', 'SU.TO', 'TRP.TO',
            'BCE.TO', 'MFC.TO', 'ATD.TO', 'CNQ.TO', 'FTS.TO', 'BAM.TO', 'RCI-B.TO', 'T.TO', 'QSR.TO', 'GWO.TO',
            'ABX.TO', 'PPL.TO', 'EMA.TO', 'L.TO', 'CVE.TO', 'POW.TO', 'SLF.TO', 'IMO.TO', 'IFC.TO', 'WCN.TO',
            'CCO.TO', 'K.TO', 'TRI.TO', 'GIB-A.TO', 'FFH.TO', 'CSU.TO', 'DOL.TO', 'SHOP.TO', 'NTR.TO', 'QBR-B.TO'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with TSX constituents: {str(e)}")
        return []

def get_bovespa_constituents():
    """Get Bovespa constituents (Brazil).
    Note: This returns a subset of prominent Bovespa constituents."""
    try:
        symbols = [
            'VALE3.SA', 'ITUB4.SA', 'PETR4.SA', 'BBDC4.SA', 'B3SA3.SA', 'ABEV3.SA', 'WEGE3.SA', 'RENT3.SA', 'BBAS3.SA', 'ITSA4.SA',
            'PETR3.SA', 'SUZB3.SA', 'EQTL3.SA', 'JBSS3.SA', 'RADL3.SA', 'BBDC3.SA', 'RAIL3.SA', 'GGBR4.SA', 'MGLU3.SA', 'VIVT3.SA',
            'BRFS3.SA', 'KLBN11.SA', 'LREN3.SA', 'EMBR3.SA', 'BPAC11.SA', 'CSNA3.SA', 'CMIG4.SA', 'ELET3.SA', 'CSAN3.SA', 'CIEL3.SA',
            'HYPE3.SA', 'ENEV3.SA', 'BRAP4.SA', 'AZUL4.SA', 'CRFB3.SA', 'CPLE6.SA', 'EGIE3.SA', 'IRBR3.SA', 'RRRP3.SA', 'TOTS3.SA'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with Bovespa constituents: {str(e)}")
        return []

def get_ipc_constituents():
    """Get IPC constituents (Mexico).
    Note: This returns the main stocks in the Mexican IPC index."""
    try:
        symbols = [
            'AMXL.MX', 'FEMSAUBD.MX', 'WALMEX.MX', 'GFNORTEO.MX', 'GMEXICOB.MX', 'CEMEXCPO.MX', 'TLEVISACPO.MX', 'KIMBERA.MX', 'BIMBOA.MX', 'GAPB.MX',
            'ASURB.MX', 'ELEKTRA.MX', 'CUERVO.MX', 'GRUMAB.MX', 'OMAB.MX', 'ALFAA.MX', 'AC.MX', 'GENTERA.MX', 'PINFRA.MX', 'GCARSOA1.MX',
            'PE&OLES.MX', 'GFINBURO.MX', 'ALSEA.MX', 'MEXCHEM.MX', 'LABB.MX', 'LIVEPOLC.MX', 'BOLSAA.MX', 'ICHB.MX', 'SITESB.MX', 'VESTA.MX'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with IPC constituents: {str(e)}")
        return []

def get_aex_constituents():
    """Get AEX constituents (Netherlands)."""
    try:
        symbols = [
            'ADYEN.AS', 'ASML.AS', 'AD.AS', 'AKZA.AS', 'MT.AS', 'ASRNL.AS', 'DSM.AS', 'HEIA.AS', 'IMCD.AS', 'ING.AS',
            'KPN.AS', 'NN.AS', 'PHIA.AS', 'RAND.AS', 'REN.AS', 'RDSA.AS', 'URW.AS', 'UNA.AS', 'VPK.AS', 'WKL.AS',
            'ABN.AS', 'AGN.AS', 'ADYEN.AS', 'INGA.AS', 'TKWY.AS'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with AEX constituents: {str(e)}")
        return []

def get_bel20_constituents():
    """Get BEL 20 constituents (Belgium)."""
    try:
        symbols = [
            'ABI.BR', 'ACKB.BR', 'APAM.BR', 'ARGX.BR', 'BPOST.BR', 'COFB.BR', 'COLR.BR', 'ELI.BR', 'GBLB.BR', 'KBC.BR',
            'LOTB.BR', 'PROX.BR', 'SOF.BR', 'SOLB.BR', 'TNET.BR', 'UCB.BR', 'UMI.BR', 'WDP.BR', 'AED.BR', 'GLPG.BR'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with BEL 20 constituents: {str(e)}")
        return []

def get_omx30_constituents():
    """Get OMX Stockholm 30 constituents (Sweden)."""
    try:
        symbols = [
            'ABB.ST', 'ALFA.ST', 'ALIV-SDB.ST', 'ASSA-B.ST', 'ATCO-A.ST', 'ATCO-B.ST', 'AXFO.ST', 'BOL.ST', 'ELUX-B.ST', 'ERIC-B.ST',
            'ESSITY-B.ST', 'EVO.ST', 'GETI-B.ST', 'HM-B.ST', 'HEXA-B.ST', 'INVE-B.ST', 'KINV-B.ST', 'NDA-SE.ST', 'SAND.ST', 'SCA-B.ST',
            'SEB-A.ST', 'SHB-A.ST', 'SINCH.ST', 'SKA-B.ST', 'SKF-B.ST', 'SWED-A.ST', 'SWMA.ST', 'TEL2-B.ST', 'TELIA.ST', 'VOLV-B.ST'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with OMX Stockholm 30 constituents: {str(e)}")
        return []

def get_wig20_constituents():
    """Get WIG20 constituents (Poland)."""
    try:
        symbols = [
            'ALE.WA', 'CCC.WA', 'CDR.WA', 'CPS.WA', 'DNP.WA', 'JSW.WA', 'KGH.WA', 'LPP.WA', 'LTS.WA', 'MBK.WA',
            'OPL.WA', 'PEO.WA', 'PGE.WA', 'PGN.WA', 'PKN.WA', 'PKO.WA', 'PLY.WA', 'PZU.WA', 'SPL.WA', 'TPE.WA'
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error with WIG20 constituents: {str(e)}")
        return []

def filter_against_valid_tickers(all_symbols):
    """
    Filter symbols against valid tickers list from yfinance.csv.
    
    Args:
        all_symbols (list): List of all constituent symbols
        
    Returns:
        list: Filtered list of valid symbols
    """
    try:
        # Check if yfinance.csv exists
        input_dir = Path(__file__).parent / 'input'
        yfinance_path = input_dir / 'yfinance.csv'
        
        if not yfinance_path.exists():
            logger.warning(f"Valid tickers file not found: {yfinance_path}")
            logger.warning("All tickers will be included. Run 'python -m yahoofinance.validate' to filter invalid tickers.")
            return sorted(set(all_symbols))
            
        # Load valid tickers
        valid_df = pd.read_csv(yfinance_path)
        if 'symbol' not in valid_df.columns:
            logger.warning("Symbol column not found in valid tickers file")
            return sorted(set(all_symbols))
            
        valid_tickers = set(valid_df['symbol'].tolist())
        
        # Filter tickers
        filtered_symbols = [symbol for symbol in all_symbols if symbol in valid_tickers]
        
        logger.info(f"Filtered {len(all_symbols)} constituents to {len(filtered_symbols)} valid tickers")
        return sorted(set(filtered_symbols))
    except Exception as e:
        logger.error(f"Error filtering constituents: {str(e)}")
        # Fallback to all symbols
        return sorted(set(all_symbols))

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
        
        # Filter against valid tickers
        filtered_symbols = filter_against_valid_tickers(all_symbols)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame({'symbol': filtered_symbols})
        filepath = input_dir / 'cons.csv'
        df.to_csv(filepath, index=False)
        logger.info(f"Successfully saved {len(filtered_symbols)} filtered constituents to {filepath}")
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
    
    logger.info("Fetching NASDAQ-100 constituents")
    all_symbols.extend(get_nasdaq100_constituents())
    
    logger.info("Fetching Dow Jones constituents")
    all_symbols.extend(get_dowjones_constituents())
    
    logger.info("Fetching Russell 2000 constituents")
    all_symbols.extend(get_russell2000_constituents())
    
    # European Indices
    logger.info("Fetching FTSE 100 constituents")
    all_symbols.extend(get_ftse100_constituents())
    
    logger.info("Fetching CAC 40 constituents")
    all_symbols.extend(get_cac40_constituents())
    
    logger.info("Fetching DAX constituents")
    all_symbols.extend(get_dax_constituents())
    
    logger.info("Fetching IBEX 35 constituents")
    all_symbols.extend(get_ibex_constituents())
    
    logger.info("Fetching FTSE MIB constituents")
    all_symbols.extend(get_ftsemib_constituents())
    
    logger.info("Fetching PSI constituents")
    all_symbols.extend(get_psi_constituents())
    
    logger.info("Fetching SMI constituents")
    all_symbols.extend(get_smi_constituents())
    
    logger.info("Fetching OMXC25 constituents")
    all_symbols.extend(get_omxc25_constituents())
    
    logger.info("Fetching ATHEX constituents")
    all_symbols.extend(get_athex_constituents())
    
    logger.info("Fetching AEX constituents")
    all_symbols.extend(get_aex_constituents())
    
    logger.info("Fetching BEL 20 constituents")
    all_symbols.extend(get_bel20_constituents())
    
    logger.info("Fetching OMX Stockholm 30 constituents")
    all_symbols.extend(get_omx30_constituents())
    
    logger.info("Fetching WIG20 constituents")
    all_symbols.extend(get_wig20_constituents())
    
    # Asian Indices
    logger.info("Fetching Nikkei 225 constituents")
    all_symbols.extend(get_nikkei225_constituents())
    
    logger.info("Fetching Hang Seng constituents")
    all_symbols.extend(get_hangseng_constituents())
    
    logger.info("Fetching CSI 300 constituents")
    all_symbols.extend(get_csi300_constituents())
    
    logger.info("Fetching KOSPI constituents")
    all_symbols.extend(get_kospi_constituents())
    
    logger.info("Fetching NIFTY 50 constituents")
    all_symbols.extend(get_nifty50_constituents())
    
    logger.info("Fetching ASX 200 constituents")
    all_symbols.extend(get_asx200_constituents())
    
    logger.info("Fetching TAIEX constituents")
    all_symbols.extend(get_taiex_constituents())
    
    # Americas (non-US)
    logger.info("Fetching TSX constituents")
    all_symbols.extend(get_tsx_constituents())
    
    logger.info("Fetching Bovespa constituents")
    all_symbols.extend(get_bovespa_constituents())
    
    logger.info("Fetching IPC constituents")
    all_symbols.extend(get_ipc_constituents())
    
    save_constituents_to_csv(all_symbols)

if __name__ == "__main__":
    main()