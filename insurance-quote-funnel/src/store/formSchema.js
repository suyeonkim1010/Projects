import { z } from "zod";

export const formSchema = z.object({
  fullName: z.string().min(2, "Please enter your full name."),
  email: z.string().email("Please enter a valid email."),
  phone: z
    .string()
    .min(10, "Please enter a valid phone number (10+ digits).")
    .regex(/^[0-9()+\-\s]+$/, "Phone can only include numbers and symbols like +()-"),
  experience: z
    .string()
    .min(10, "Please enter at least 10 characters.")
    .max(500, "Please keep this under 500 characters.")
});

export const stepSchemas = {
  1: formSchema.pick({ fullName: true }),
  2: formSchema.pick({ email: true, phone: true }),
  3: formSchema.pick({ experience: true }),
  4: formSchema
};
