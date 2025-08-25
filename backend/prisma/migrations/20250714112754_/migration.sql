/*
  Warnings:

  - You are about to drop the column `updated_at` on the `integrations` table. All the data in the column will be lost.
  - You are about to drop the column `created_by_id` on the `tasks` table. All the data in the column will be lost.
  - You are about to drop the column `is_ai_generated` on the `tasks` table. All the data in the column will be lost.
  - You are about to drop the column `meeting_id` on the `tasks` table. All the data in the column will be lost.
  - You are about to drop the column `updated_at` on the `tenants` table. All the data in the column will be lost.
  - You are about to drop the column `updated_at` on the `users` table. All the data in the column will be lost.
  - You are about to drop the `meetings` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `realtime_sessions` table. If the table is not empty, all the data it contains will be lost.
  - A unique constraint covering the columns `[tenant_id,task_number]` on the table `tasks` will be added. If there are existing duplicate values, this will fail.
  - Added the required column `project_id` to the `tasks` table without a default value. This is not possible if the table is not empty.
  - Added the required column `task_number` to the `tasks` table without a default value. This is not possible if the table is not empty.

*/
-- CreateEnum
CREATE TYPE "InputType" AS ENUM ('VOICE', 'TEXT');

-- CreateEnum
CREATE TYPE "ProcessingStatus" AS ENUM ('RECEIVED', 'PROCESSING', 'COMPLETED', 'FAILED');

-- CreateEnum
CREATE TYPE "TaskPriority" AS ENUM ('HIGH', 'MEDIUM', 'LOW');

-- DropForeignKey
ALTER TABLE "meetings" DROP CONSTRAINT "meetings_host_id_fkey";

-- DropForeignKey
ALTER TABLE "meetings" DROP CONSTRAINT "meetings_tenant_id_fkey";

-- DropForeignKey
ALTER TABLE "realtime_sessions" DROP CONSTRAINT "realtime_sessions_host_id_fkey";

-- DropForeignKey
ALTER TABLE "realtime_sessions" DROP CONSTRAINT "realtime_sessions_meeting_id_fkey";

-- DropForeignKey
ALTER TABLE "realtime_sessions" DROP CONSTRAINT "realtime_sessions_tenant_id_fkey";

-- DropForeignKey
ALTER TABLE "tasks" DROP CONSTRAINT "tasks_created_by_id_fkey";

-- DropForeignKey
ALTER TABLE "tasks" DROP CONSTRAINT "tasks_meeting_id_fkey";

-- DropIndex
DROP INDEX "users_tenant_id_slack_user_id_key";

-- AlterTable
ALTER TABLE "integrations" DROP COLUMN "updated_at",
ADD COLUMN     "config" JSONB;

-- AlterTable
ALTER TABLE "tasks" DROP COLUMN "created_by_id",
DROP COLUMN "is_ai_generated",
DROP COLUMN "meeting_id",
ADD COLUMN     "completed_at" TIMESTAMP(3),
ADD COLUMN     "complexity" VARCHAR(20),
ADD COLUMN     "parent_id" TEXT,
ADD COLUMN     "priority" "TaskPriority" NOT NULL DEFAULT 'MEDIUM',
ADD COLUMN     "project_id" TEXT NOT NULL,
ADD COLUMN     "task_number" VARCHAR(20) NOT NULL;

-- AlterTable
ALTER TABLE "tenants" DROP COLUMN "updated_at";

-- AlterTable
ALTER TABLE "users" DROP COLUMN "updated_at",
ADD COLUMN     "available_hours" DOUBLE PRECISION DEFAULT 40,
ADD COLUMN     "experience_level" VARCHAR(20) DEFAULT 'junior',
ADD COLUMN     "jira_user_id" VARCHAR(100),
ADD COLUMN     "last_assigned_at" TIMESTAMP(3),
ADD COLUMN     "preferred_types" JSONB,
ADD COLUMN     "skills" JSONB;

-- DropTable
DROP TABLE "meetings";

-- DropTable
DROP TABLE "realtime_sessions";

-- DropEnum
DROP TYPE "SessionStatus";

-- DropEnum
DROP TYPE "SttStatus";

-- CreateTable
CREATE TABLE "slack_inputs" (
    "id" TEXT NOT NULL,
    "tenant_id" TEXT NOT NULL,
    "slack_channel_id" VARCHAR(50) NOT NULL,
    "slack_user_id" VARCHAR(50) NOT NULL,
    "input_type" "InputType" NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "content" TEXT NOT NULL,
    "status" "ProcessingStatus" NOT NULL DEFAULT 'RECEIVED',

    CONSTRAINT "slack_inputs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "projects" (
    "id" TEXT NOT NULL,
    "tenant_id" TEXT NOT NULL,
    "slack_input_id" TEXT NOT NULL,
    "title" VARCHAR(200) NOT NULL,
    "overview" TEXT NOT NULL,
    "content" JSONB NOT NULL,
    "notion_page_url" TEXT,
    "notion_status" VARCHAR(20),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "projects_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "task_metadata" (
    "id" TEXT NOT NULL,
    "task_id" TEXT NOT NULL,
    "estimated_hours" DOUBLE PRECISION,
    "actual_hours" DOUBLE PRECISION,
    "required_skills" JSONB,
    "task_type" VARCHAR(50),
    "assignment_score" DOUBLE PRECISION,
    "assignment_reason" TEXT,
    "jira_issue_key" VARCHAR(50),
    "jira_status" VARCHAR(20),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "task_metadata_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "task_assignment_logs" (
    "id" TEXT NOT NULL,
    "task_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "assigned_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "assignment_score" DOUBLE PRECISION NOT NULL,
    "score_breakdown" JSONB NOT NULL,
    "reason" TEXT NOT NULL,
    "alternatives" JSONB,
    "algorithm_version" VARCHAR(10) NOT NULL DEFAULT '1.0',

    CONSTRAINT "task_assignment_logs_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "task_metadata_task_id_key" ON "task_metadata"("task_id");

-- CreateIndex
CREATE UNIQUE INDEX "tasks_tenant_id_task_number_key" ON "tasks"("tenant_id", "task_number");

-- AddForeignKey
ALTER TABLE "slack_inputs" ADD CONSTRAINT "slack_inputs_tenant_id_fkey" FOREIGN KEY ("tenant_id") REFERENCES "tenants"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "projects" ADD CONSTRAINT "projects_slack_input_id_fkey" FOREIGN KEY ("slack_input_id") REFERENCES "slack_inputs"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "projects" ADD CONSTRAINT "projects_tenant_id_fkey" FOREIGN KEY ("tenant_id") REFERENCES "tenants"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "tasks" ADD CONSTRAINT "tasks_parent_id_fkey" FOREIGN KEY ("parent_id") REFERENCES "tasks"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "tasks" ADD CONSTRAINT "tasks_project_id_fkey" FOREIGN KEY ("project_id") REFERENCES "projects"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "task_metadata" ADD CONSTRAINT "task_metadata_task_id_fkey" FOREIGN KEY ("task_id") REFERENCES "tasks"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "task_assignment_logs" ADD CONSTRAINT "task_assignment_logs_task_id_fkey" FOREIGN KEY ("task_id") REFERENCES "tasks"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "task_assignment_logs" ADD CONSTRAINT "task_assignment_logs_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
