import { format, isToday, parseISO } from 'date-fns';
import { ko } from 'date-fns/locale';

export const formatDate = (date: string | Date): string => {
  const parsedDate = typeof date === 'string' ? parseISO(date) : date;
  
  if (isToday(parsedDate)) {
    return '오늘';
  }
  
  return format(parsedDate, 'MM.dd', { locale: ko });
};

export const formatDateTime = (date: string | Date): string => {
  const parsedDate = typeof date === 'string' ? parseISO(date) : date;
  return format(parsedDate, 'yyyy.MM.dd HH:mm', { locale: ko });
};

export const cn = (...classes: string[]): string => {
  return classes.filter(Boolean).join(' ');
};

export const getStatusColor = (status: string): string => {
  switch (status) {
    case 'completed':
      return 'text-green-500';
    case 'inProgress':
      return 'text-blue-500';
    case 'scheduled':
      return 'text-orange-500';
    default:
      return 'text-gray-500';
  }
};

export const getStatusIcon = (status: string): string => {
  switch (status) {
    case 'completed':
      return '✅';
    case 'inProgress':
      return '⏱️';
    case 'scheduled':
      return '🎯';
    default:
      return '📋';
  }
}; 