import time


class Performance:
    def __init__(self, debug=True):
        self.debug = debug
        self.performances = {}

    def tick(self, key, force_print=False):
        if self.debug or force_print:
            if key in self.performances.keys():
                print('Warning you ary measure same functions twice for {} function'.format(key))
                return
            self.performances[key] = time.time()

    def end(self, key, force_print=False):
        if self.debug or force_print:
            if key not in self.performances.keys():
                print('Warning you ary measure {} function without use Performance.tick() first'.format(key))
                return
            end_time = (time.time() - self.performances[key]) * 1000
            del self.performances[key]
            print("{} take {} MS".format(key, end_time))
