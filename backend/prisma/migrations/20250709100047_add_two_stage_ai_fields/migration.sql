-- AlterTable
ALTER TABLE "meetings" ADD COLUMN     "generated_tasks" JSONB,
ADD COLUMN     "notion_project" JSONB,
ADD COLUMN     "task_master_prd" JSONB;
