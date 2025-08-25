import { Volume2, Mic, FileText, Settings } from 'lucide-react';

const RightSidebar = () => {
  return (
    <div className="w-80 bg-white shadow-lg p-6">
      {/* 바로가기 */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-4">바로가기</h3>
        <div className="grid grid-cols-2 gap-2">
          <button className="bg-gray-200 hover:bg-gray-300 p-3 rounded-lg text-sm text-center transition-colors">
            <Volume2 className="w-6 h-6 mx-auto mb-1" />
            <div>음성 과제 실행</div>
          </button>
          <button className="bg-gray-200 hover:bg-gray-300 p-3 rounded-lg text-sm text-center transition-colors">
            <Mic className="w-6 h-6 mx-auto mb-1" />
            <div>실시간 녹음</div>
          </button>
          <button className="bg-gray-200 hover:bg-gray-300 p-3 rounded-lg text-sm text-center transition-colors">
            <FileText className="w-6 h-6 mx-auto mb-1" />
            <div>업무 관리</div>
          </button>
          <button className="bg-gray-200 hover:bg-gray-300 p-3 rounded-lg text-sm text-center transition-colors">
            <Settings className="w-6 h-6 mx-auto mb-1" />
            <div>내 업무 보기</div>
          </button>
        </div>
      </div>

      {/* 담당자 정보 */}
      <div>
        <h3 className="text-lg font-semibold mb-4">담당자</h3>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">진행중인 업무</span>
            <span className="text-gray-500">마감일</span>
          </div>
          <div className="border-b pb-2"></div>
          
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="bg-gray-100 p-2 rounded">
              <div className="text-xs text-gray-600">택스트</div>
            </div>
            <div className="bg-gray-100 p-2 rounded">
              <div className="text-xs text-gray-600">택스트</div>
            </div>
            <div className="bg-gray-100 p-2 rounded">
              <div className="text-xs text-gray-600">택스트</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RightSidebar; 