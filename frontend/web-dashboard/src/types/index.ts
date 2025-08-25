export interface Task {
  id: string;
  title: string;
  description?: string;
  status: 'TODO' | 'IN_PROGRESS' | 'DONE';
  assigneeId?: string;
  assignee?: User;
  dueDate?: string;
  priority: 'HIGH' | 'MEDIUM' | 'LOW';
  complexity?: string;
  metadata?: TaskMetadata;
  children?: Task[];
}

export interface User {
  id: string;
  name: string;
  email: string;
  role: 'OWNER' | 'ADMIN' | 'MEMBER';
  skills?: string[];
  availableHours?: number;
  experienceLevel?: string;
}

export interface Project {
  id: string;
  title: string;
  overview: string;
  content: any;
  notionPageUrl?: string;
  createdAt: string;
  tasks: Task[];
}


export interface TaskMetadata {
  estimatedHours?: number;
  actualHours?: number;
  requiredSkills?: string[];
  taskType?: string;
  jiraIssueKey?: string;
  assignmentScore?: number;
  assignmentReason?: string;
  jiraStatus?: string;
}


export interface MeetingFilters {
  meetingName: string;
  date: string;
  participants: string;
}

export interface TaskFilters {
  assignee: string;
  status: string;
  priority: string;
  search: string;
}

export interface Notification {
  id: string;
  message: string;
  type: 'warning' | 'info' | 'error';
}

export interface RecentSummary {
  id: string;
  title: string;
  date: string;
  taskCount: number;
}

export interface User {
  id: string;
  name: string;
  email: string;
  role: 'OWNER' | 'ADMIN' | 'MEMBER';
  skills?: string[];
  availableHours?: number;
  experienceLevel?: string;
}

export interface DashboardStats {
  totalMeetings: number;
  averageProcessingTime: number;
  accuracy: number;
  completedTasks: number;
  inProgressTasks: number;
  scheduledTasks: number;
}