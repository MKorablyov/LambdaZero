from ray.tune.logger import Logger, TBXLogger

class TBXLoggerWrrapper(TBXLogger):


    # todo self._on_result = self.on_result

    def _prefilter(self, result):

        # filt_result = if type(r) for key,value in result]
        # return filt_result
        pass

    def on_result(self,result):
        # todo: filt_result = self._pre_filter(result)
        # todo self._on_result(filt_result)
        pass

    # todo super(TBXLogger.__init__())


# from ray.tune.logger import Logger
# class SomeLogger(Logger):
#     def _init(self):
#         print("init logger 111111111111111111111111111111111")
#         time.sleep(1)
#         pass
#
#     def on_result(self, result):
#         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!1", result)
#
#     def write(self, b):
#         print("11111111111111 b ", b)
#
#     def close(self):
#         pass