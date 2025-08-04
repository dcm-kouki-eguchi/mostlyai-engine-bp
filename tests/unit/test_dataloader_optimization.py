# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
from unittest.mock import patch

from mostlyai.engine._tabular.training import get_optimal_num_workers


class TestDataLoaderOptimization(unittest.TestCase):
    """DataLoader最適化機能のテスト"""

    @patch('psutil.cpu_count')
    @patch('os.cpu_count')
    def test_get_optimal_num_workers_normal(self, mock_os_cpu_count, mock_psutil_cpu_count):
        """正常なCPU環境でのワーカー数決定テスト"""
        mock_os_cpu_count.return_value = 8  # 論理コア数
        mock_psutil_cpu_count.return_value = 4  # 物理コア数
        
        result = get_optimal_num_workers()
        
        # 物理コア数の50%、最大4に制限
        expected = min(4, max(1, 4 // 2))  # = 2
        self.assertEqual(result, expected)

    @patch('psutil.cpu_count')
    @patch('os.cpu_count')
    def test_get_optimal_num_workers_high_core_count(self, mock_os_cpu_count, mock_psutil_cpu_count):
        """高コア数環境でのワーカー数決定テスト"""
        mock_os_cpu_count.return_value = 16  # 論理コア数
        mock_psutil_cpu_count.return_value = 8  # 物理コア数
        
        result = get_optimal_num_workers()
        
        # 物理コア数の50%だが最大4に制限される
        expected = min(4, max(1, 8 // 2))  # = 4
        self.assertEqual(result, expected)

    @patch('psutil.cpu_count')
    @patch('os.cpu_count')
    def test_get_optimal_num_workers_low_core_count(self, mock_os_cpu_count, mock_psutil_cpu_count):
        """低コア数環境でのワーカー数決定テスト"""
        mock_os_cpu_count.return_value = 2  # 論理コア数
        mock_psutil_cpu_count.return_value = 1  # 物理コア数
        
        result = get_optimal_num_workers()
        
        # 最小1に制限される
        expected = min(4, max(1, 1 // 2))  # = 1
        self.assertEqual(result, expected)

    @patch('psutil.cpu_count')
    @patch('os.cpu_count')
    def test_get_optimal_num_workers_psutil_error(self, mock_os_cpu_count, mock_psutil_cpu_count):
        """psutil利用不可時のフォールバックテスト"""
        mock_os_cpu_count.return_value = 8  # 論理コア数
        mock_psutil_cpu_count.side_effect = Exception("psutil not available")
        
        result = get_optimal_num_workers()
        
        # フォールバック: physical_cores = cpu_cores // 2
        physical_cores_fallback = 8 // 2  # = 4
        expected = min(4, max(1, physical_cores_fallback // 2))  # = 2
        self.assertEqual(result, expected)

    @patch('psutil.cpu_count')
    @patch('os.cpu_count')
    def test_get_optimal_num_workers_edge_cases(self, mock_os_cpu_count, mock_psutil_cpu_count):
        """エッジケースのテスト"""
        # ケース1: 物理コア数が0の場合
        mock_os_cpu_count.return_value = 4
        mock_psutil_cpu_count.return_value = 0
        
        result = get_optimal_num_workers()
        expected = min(4, max(1, 0 // 2))  # = 1
        self.assertEqual(result, expected)
        
        # ケース2: 非常に高いコア数
        mock_psutil_cpu_count.return_value = 32
        
        result = get_optimal_num_workers()
        expected = min(4, max(1, 32 // 2))  # = 4 (最大値で制限)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
