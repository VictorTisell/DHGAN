import pandas as pd
import numpy as np
import QuantLib as ql
import pandas_datareader as pdr
import datetime

class Loader:
    def __init__(self, symbol = '^GSPC', symbol_list = None, start=datetime.date(2000, 1, 1),
                end = datetime.date(2020,9,11), seq_len = 31, standard = False):
        self.symbol = symbol
        self.symbol_list = symbol_list
        self.start = start
        self.end = end
        self.seq_len = seq_len
        self.df, self.starting_values = self._load_data()
        self.data, self.norm_data, self.min_val, self.max_val = self._preprocess_data_normalize_first_entry()
    def _load_data(self):
        """ Data importer
        Args: Symbol
        Returns: Dataframe containing close prices
        """
        if self.symbol_list is not None:
            data = pd.DataFrame()
            try:
                for ticker in self.symbol_list:
                    df = pdr.get_data_yahoo(ticker, self.start, self.end)['Close']
                    df.dropna(inplace = True)
                    df.rename('{}'.format(ticker), axis = 1, inplace = True)
                    data = data.join(df, how = 'outer')
                    data.dropna(inplace = True)
            except:
                raise RuntimeError(f"Could not download data for {self.symbol} from {self.start} to {self.end}.")
        else:
            try:
                data = pdr.get_data_yahoo(self.symbol, self.start, self.end)
            except:
                raise RuntimeError(f"Could not download data for {self.symbol} from {self.start} to {self.end}.")
        starting_values = data.tail(1)
        data = np.log(data)
        return data, starting_values
    def _preprocess_data(self):
        """
        Preprocesses original data
        """
        data = self.df.to_numpy()
        # Convert to Chronological data
        data = data[::-1]
        # data = _MinMaxScaler(data)
        temp_data = []
        for i in range(0, len(data)-self.seq_len):
            _x = data[i:i+seq_len]
            temp_data.append(_x)
        # OPtional Create I.I.D sequences
        outdata = temp_data
        idx = np.random.permutation(len(temp_data))
        outdata = []
        for i in range(len(temp_data)):
            outdata.append(temp_data[idx[i]])

        return outdata
        # self.data = _preprocess_data(self)

    def _preprocess_data_normalize_first_entry(self):
        data = self.df.to_numpy()
        # Convert to Chronological data
        # data = data[::-1]
        # data, min_val, max_val = MinMaxScaler(data)
        temp_data = []
        for i in range(0, len(data)-self.seq_len):
            _x = data[i:i+self.seq_len]
            starting_values = _x[0,:]
            x = _x-starting_values
            temp_data.append(x)
        outdata = temp_data
        idx = np.random.permutation(len(temp_data))
        outdata = []
        for i in range(len(temp_data)):
            outdata.append(temp_data[idx[i]])
        norm_data, min_val, max_val = self.MinMaxScaler(np.asarray(outdata))

        return outdata, norm_data, min_val, max_val
    def MinMaxScaler(self, data):
        min_val = np.min(np.min(data, axis = 0), axis = 0)
        data = data - min_val

        max_val = np.max(np.max(data, axis = 0), axis = 0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val
class QL_Helpers:
    def dt_to_ql(self, date):
        return ql.Date(date.day, date.month, date.year)
    def ql_to_dt(self, Dateql):
        return datetime.datetime(Dateql.year(), Dateql.month(), Dateql.dayOfMonth())

    def ql_to_dt_settings_dict(Datesql, dictionary):
        if bool(dictionary):
            for dql in Datesql:
                helper = ql_to_dt(dql)
                dictionary[helper] = dictionary.pop(dql)

    def dt_to_ql_settings_dict(dates, dictionary):
        if bool(dictionary):
            dates = sorted(dates)
            for date in dates:
                helper = dt_to_ql(date)
                dictionary[helper] = dictionary.pop(date)
    def set_rf(calc_date, rf, day_count):
        return ql.YieldTermStructureHandle(ql.FlatForward(calc_date, rf, day_count))
    def set_dividend(calc_date, dividend_rate, day_count):
        return ql.YieldTermStructureHandle(
                ql.FlatForward(calc_date, dividend_rate, day_count))
    def set_spot(spot_price):
        return ql.QuoteHandle(ql.SimpleQuote(spot_price))
    def EU_option(calc_date, option_type, strike, ttm):
        cashflow = ql.PlainVanillaPayoff(option_type, strike)
        maturity = calc_date + int(ttm)
        ex = ql.EuropeanExercise(maturity)
        return ql.VanillaOption(cashflow, ex)

    def year_fraction(date, ttm):
        return [day_count.yearFraction(date, date + int(nd)) for nd in ttm]
class Option_data(QL_Helpers):
    def __init__(self, option_data, option_settings_dict):
        super().__init__()
        self.option_data = option_data
        self.option_settings = option_settings_dict
        self.start_dt = self.option_settings['start']
        self.start_ql = self.dt_to_ql(self.start_dt)
        self.day_count = self.option_settings['day_count']
        self.calendar = self.option_settings['calendar']
        self.strikes = self.option_settings['strikes']
        self.maturities = self.option_settings['maturities']
        ql.Settings.instance().evaluationDate = self.start_ql
        self.preprocessed_chain, self.price_array, self.iv_array = self._preprocess()

    def _preprocess(self):
        df = self.option_data
        tmp =[self.ql_to_dt(self.start_ql + ttm) for ttm in self.maturities]
        date_format = '%d/%m/%Y'
        df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])
        df.dropna(inplace = True)
        df = df.loc[(df['Strike'].isin(self.strikes)) & (df['Expiration Date'].isin(tmp))]
        df.set_index('Expiration Date', drop = True,inplace = True)
        iv_array = np.asarray(df[['Strike', 'IV']])
        df['ttm'] = df.index - self.ql_to_dt(self.start_ql)
        df['ttm'] = df['ttm'].dt.days
        sub_array = np.asarray(df[['Strike','ttm', 'Ask']])
        return df, sub_array, iv_array
    def GBM(S0, r, sigma, timesteps, dt ,nb_sims, nb_assets, corr):
        features = np.zeros(shape = (nb_sims, timesteps, nb_assets))
        features[:,0, :] = S0
        pass


if __name__ == '__main__':
    datafolder_path = 'datafolder'
    filename = 'quotedata_SPX.csv'
    filepath = datafolder_path + '/' + filename
    data = pd.read_csv(filepath, sep = ';')
    calendar = ql.UnitedStates()
    Ks = [3000, 3100, 3200, 3300, 3400]
    ttms = [10, 21, 31, 62]
    settings = {'start': datetime.date(2020,9,11),
                'day_count': ql.Actual365Fixed(),
                'calendar': ql.UnitedStates(),
                'strikes': Ks,
                'maturities':ttms}
    preprocessor = Option_data(data, settings)
    symbol_list = ['^GSPC', '^VIX', '^NDX', '^RUT', '^DJI',
                '^FVX','^TNX', '^TYX', 'EURUSD=X']
    data_ = Loader(symbol_list = symbol_list)
    print(data_.df)
