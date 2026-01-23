import { z } from "zod";

export const quoteSchema = z.object({
  // Step 1
  fullName: z.string().min(2, "Please enter your full name."),
  email: z.string().email("Please enter a valid email."),
  phone: z
    .string()
    .min(10, "Please enter a valid phone number (10+ digits).")
    .regex(/^[0-9()+\-\s]+$/, "Phone can only include numbers and symbols like +()-"),

  // Step 2
  vehicleYear: z
    .string()
    .regex(/^\d{4}$/, "Enter a 4-digit year.")
    .refine((v) => {
      const year = Number(v);
      return year >= 1980 && year <= new Date().getFullYear() + 1;
    }, "Enter a reasonable year."),
  vehicleMake: z.string().min(2, "Enter the vehicle make."),
  vehicleModel: z.string().min(1, "Enter the vehicle model."),

  // Step 3
  coverageType: z.enum(["basic", "standard", "premium"], {
    errorMap: () => ({ message: "Please select a coverage type." })
  }),
  hasAccidents: z.enum(["yes", "no"], {
    errorMap: () => ({ message: "Please select yes/no." })
  })
});

export type QuoteFormData = z.infer<typeof quoteSchema>;
