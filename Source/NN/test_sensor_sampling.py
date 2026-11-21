"""Raw invalid distances must not be treated as genuine near measurements."""
import json
import unittest
from analyze_sensor_sampling import analyze


class SensorSamplingTests(unittest.TestCase):
    def test_status_and_distance_are_both_needed(self):
        frame=dict(raw_mm=[500]*64,status=[5]*64,targets=[1]*64,sigma_mm=[10]*64,
                   signal_kcps_spad=[20]*64,ambient_kcps_spad=[1]*64)
        frame['status'][0]=255
        frame['raw_mm'][1]=2000
        frame['status'][2]=9
        frame['raw_mm'][3]=-1
        result=analyze('sensor_diag '+json.dumps(frame)+'\n')
        self.assertEqual(result['snapshot_count'],1)
        self.assertEqual(result['snapshot_valid_zones']['median'],62)
        self.assertEqual(result['snapshot_valid_at_most_1m_zones']['median'],61)
        self.assertEqual(result['rejected_untrusted_raw_values_at_most_1m_by_status'],{255:1})
        self.assertEqual(result['rejected_status_histogram'],{5:1,255:1})

    def test_truncated_snapshot_is_not_accepted_as_full_data(self):
        result=analyze('sensor_diag {"raw_mm":[1]}\nsensor_diag {bad json\n')
        self.assertEqual(result['snapshot_count'],0)
        self.assertEqual(len(result['malformed']),2)
        self.assertIsNone(result['snapshot_valid_zones'])


if __name__=='__main__':
    unittest.main()
